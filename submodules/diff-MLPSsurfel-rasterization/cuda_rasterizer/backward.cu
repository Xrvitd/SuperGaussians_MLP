/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
// #include "mlp.h"
// #include "common.h"
namespace cg = cooperative_groups;


__device__ glm::vec4 MySigmoid(glm::vec4 x)
{
	return glm::vec4(1.0f / (1.0f + exp(-x.x)), 1.0f / (1.0f + exp(-x.y)), 1.0f / (1.0f + exp(-x.z)), 1.0f / (1.0f + exp(-x.w)));
}

__device__ glm::vec4 dL_dSigmoid(glm::vec4 x)
{
	return glm::vec4(1.0f / (1.0f + exp(-x.x)) * (1.0f - 1.0f / (1.0f + exp(-x.x))), 
					 1.0f / (1.0f + exp(-x.y)) * (1.0f - 1.0f / (1.0f + exp(-x.y))), 
					 1.0f / (1.0f + exp(-x.z)) * (1.0f - 1.0f / (1.0f + exp(-x.z))), 
					 1.0f / (1.0f + exp(-x.w)) * (1.0f - 1.0f / (1.0f + exp(-x.w))));
}
__device__ glm::vec4 MyRelu(glm::vec4 x)
{
	return glm::vec4(x.x > 0.0f? x.x : 0.0f,  x.y > 0.0f? x.y : 0.0f,  x.z > 0.0f? x.z : 0.0f,  x.w > 0.0f? x.w : 0.0f);
}
__device__ glm::vec4 dL_dRelu(glm::vec4 x)
{
	return glm::vec4(x.x > 0.0f? 1.0f : 0.0f,  x.y > 0.0f? 1.0f : 0.0f,  x.z > 0.0f? 1.0f : 0.0f,  x.w > 0.0f? 1.0f : 0.0f);
}

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs, float2* wichzone)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * (34);

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * (34);

	// float proj_u = wichzone[idx].x;
	// float proj_v = wichzone[idx].y;
	// // printf("Backward proj_u: %f, proj_v: %f\n", proj_u, proj_v);
	// if(isnan(proj_u) || isnan(proj_v))
	// {
	// 	printf("Backward proj_u: %f, proj_v: %f\n", proj_u, proj_v);
	// 	proj_u = 0.5;
	// 	proj_v = 0.5;
	// }
	// dL_dsh[16] = (1.0f-proj_u) * (1.0f - proj_v) * dL_dRGB;
	// dL_dsh[17] = (1.0f-proj_u) * proj_v * dL_dRGB;
	// dL_dsh[18] = proj_u * (1.0f - proj_v) * dL_dRGB;
	// dL_dsh[19] = proj_u * proj_v * dL_dRGB;
	// dL_dsh[16] =  dL_dRGB;
	// dL_dsh[17] = dL_dRGB;
	// dL_dsh[18] =  dL_dRGB;
	// dL_dsh[19] =dL_dRGB;
	// printf("Backward proj_u: %f, proj_v: %f | dL_dRGB: %f, %f, %f\n", proj_u, proj_v, dL_dRGB.x, dL_dRGB.y, dL_dRGB.z);


	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float4* __restrict__ my_opacity,
	const float* __restrict__ transMats,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dtransMat,
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	const float* __restrict__ shs,
	float2* __restrict__ wichzone,
	float* __restrict__ dL_dshs)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float4 collected_my_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	// __shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];

#if RENDER_AXUTILITY
	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;
	float dL_dmax_dweight;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];

		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
		// dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	float last_dL_dT = 0;
#endif

	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_my_opacity[block.thread_rank()] = my_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
				// collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// compute ray-splat intersection as before
			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

			// compute intersection and depth
			float rho = min(rho3d, rho2d);
			float c_d = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
			if (c_d < near_n) continue;
			float4 nor_o = collected_normal_opacity[j];
			float4 my_o = collected_my_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			// float opa = nor_o.w; // old 2dgs
			float proj_u = s.x;
			float proj_v = s.y;
			if(rho3d>rho2d)
			{
				proj_u = d.x;
				proj_v = d.y;
			}

			glm::vec3* sh = ((glm::vec3*)shs) + collected_id[j] * (16+18);

			glm::mat2x4 input = glm::mat2x4(
				glm::vec4(sh[16+0].x, sh[16+0].y, sh[16+0].z, sh[16+1].x),
				glm::vec4(sh[16+1].y, sh[16+1].z, sh[16+2].x, sh[16+2].y)
			);
			glm::vec4 input_biases = glm::vec4(sh[16+2].z, sh[16+3].x, sh[16+3].y, sh[16+3].z);
			glm::mat4x4 middle = glm::mat4x4(
				glm::vec4(sh[16+4].x, sh[16+4].y, sh[16+4].z, sh[16+5].x),
				glm::vec4(sh[16+5].y, sh[16+5].z, sh[16+6].x, sh[16+6].y),
				glm::vec4(sh[16+6].z, sh[16+7].x, sh[16+7].y, sh[16+7].z),
				glm::vec4(sh[16+8].x, sh[16+8].y, sh[16+8].z, sh[16+9].x)
			);
			glm::vec4 middle_biases = glm::vec4(sh[16+9].y, sh[16+9].z, sh[16+10].x, sh[16+10].y);
			glm::mat4x4 output = glm::mat4x4(
				glm::vec4(sh[16+10].z, sh[16+11].x, sh[16+11].y, sh[16+11].z),
				glm::vec4(sh[16+12].x, sh[16+12].y, sh[16+12].z, sh[16+13].x),
				glm::vec4(sh[16+13].y, sh[16+13].z, sh[16+14].x, sh[16+14].y),
				glm::vec4(sh[16+14].z, sh[16+15].x, sh[16+15].y, sh[16+15].z)
			);
			glm::vec4 output_biases = glm::vec4(sh[16+16].x, sh[16+16].y, sh[16+16].z, sh[16+17].x);

			// forward
			glm::vec2 input_data2 = glm::vec2(proj_u, proj_v);
			input_data2.x = 1.0f / (1.0f + exp(-proj_u));
			input_data2.y = 1.0f / (1.0f + exp(-proj_v));

			glm::vec4 L1out = input  * input_data2  + input_biases;
			glm::vec4 L1outR = MySigmoid(L1out);
			glm::vec4 L2out = middle * L1outR + middle_biases;
			glm::vec4 L2outR = MySigmoid(L2out);
			glm::vec4 L3out = output * L2outR  + output_biases;
			glm::vec4 L3outR = MySigmoid(L3out);
			// float x[3] = {L3outR.x, L3outR.y, L3outR.z};
			float x[3] = {L3outR.x*2.0f -1.0f, L3outR.y*2.0f -1.0f, L3outR.z*2.0f -1.0f};


		
			// float my_opa = my_o.x;
			float my_opa = my_o.x + 0.5 * (L3outR.w * 2.0 - 1.0);
			// float my_opa = L3outR.w;
			
			// accumulations

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			float dalpha_dgsscale = 0.0f;
			dalpha_dgsscale = my_opa * power * exp(power);
			const float alpha = min(0.99f, my_opa * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float w = alpha * T;
			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			
			// float proj_u = s.x;
			// float proj_v = s.y;
			
			

			float Myalpha = 1.0;
			float nncolor[3];
			nncolor[0] = x[0];
			nncolor[1] = x[1];
			nncolor[2] = x[2];

			




			float dl_dc[3],dc_dsx[3]={0},dc_dsy[3],dop_dsx=0.0,dop_dsy=0.0;
			float dsx_dtu[3],dsx_dtv[3],dsy_dtu[3],dsy_dtv[3],dsx_dtw[3],dsy_dtw[3];


			dsx_dtu[0] = p.x * (-1.0) / (p.z * p.z) * l.y * (-1.0);
			dsx_dtu[1] = (-1.0) / (p.z * p.z) * l.x * p.x + l.z*(-1.0) / p.z;
			dsx_dtu[2] = l.y / p.z;
			dsx_dtv[0] = p.x * (-1.0) / (p.z * p.z) * k.y;
			dsx_dtv[1] = (-1.0) / (p.z * p.z) * k.x *(-1.0) * p.x + k.z / p.z;
			dsx_dtv[2] = k.y*(-1.0) / p.z;
			dsy_dtu[0] = (-1.0) / (p.z * p.z) * l.y *(-1.0) * p.y + l.z / p.z;
			dsy_dtu[1] = p.y * (-1.0) / (p.z * p.z) * l.x;
			dsy_dtu[2] = l.y*(-1.0) / p.z;
			dsy_dtv[0] = (-1.0) / (p.z * p.z) * k.y * p.y + k.z * (-1.0) / p.z;
			dsy_dtv[1] = p.y * (-1.0) / (p.z * p.z) * k.x * (-1.0);
			dsy_dtv[2] = k.x / p.z;
			dsx_dtw[0] = p.x * (-1.0) / (p.z * p.z) * (pix.x*pix.y*Tw.y - pix.x*Tv.y - pix.x*pix.y*Tw.y + pix.y*Tu.y);
			dsx_dtw[1] = (pix.x*pix.y*Tw.z - pix.x*Tv.z - pix.x*pix.y*Tw.z + pix.y*Tu.z) / p.z + (p.x*(-1.0)/(p.z*p.z) * (pix.x*pix.y*Tw.x - pix.y*Tu.x - pix.x*pix.y*Tw.x + pix.x*Tv.x));
			dsx_dtw[2] = (pix.x*pix.y*Tw.y - pix.y*Tu.y - pix.x*pix.y*Tw.y + pix.x*Tv.y) / p.z;

			dsy_dtw[0] = (pix.x*pix.y*Tw.z - pix.y*Tu.z - pix.x*pix.y*Tw.z + pix.x*Tv.z) / p.z + (p.y*(-1.0)/(p.z*p.z) * (pix.x*pix.y*Tw.y - pix.x*Tv.y - pix.x*pix.y*Tw.y + pix.y*Tu.y));
			dsy_dtw[1] = (p.y*(-1.0)/(p.z*p.z) * (pix.x*pix.y*Tw.x - pix.y*Tu.x - pix.x*pix.y*Tw.x + pix.x*Tv.x));
			dsy_dtw[2] = (pix.x*pix.y*Tw.x - pix.x*Tv.x - pix.x*pix.y*Tw.x + pix.y*Tu.x) / p.z;
			float dl_dtu[3],dl_dtv[3],dl_dtw[3];
			for (int ch = 0; ch < C; ch++)
			{
				dl_dtu[ch] = 0.0;
				dl_dtv[ch] = 0.0;
				dl_dtw[ch] = 0.0;
			}

			for (int ch = 0; ch < C; ch++)
			{
				// const float c = collected_colors[ch * BLOCK_SIZE + j];
				const float c = collected_colors[ch * BLOCK_SIZE + j] + Myalpha * nncolor[ch];
				// const float c = Myalpha * nncolor[ch];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);

				dl_dc[ch] = dchannel_dcolor * dL_dchannel;
	
			}
		
			
		
			

			float dL_dz = 0.0f;
			float dL_dweight = 0;
#if RENDER_AXUTILITY
			const float m_d = far_n / (far_n - near_n) * (1 - near_n / c_d);
			const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * c_d * c_d);
			if (contributor == median_contributor-1) {
				dL_dz += dL_dmedian_depth;
				// dL_dweight += dL_dmax_dweight;
			}
#if DETACH_WEIGHT 
			// if not detached weight, sometimes 
			// it will bia toward creating extragated 2D Gaussians near front
			dL_dweight += 0;
#else
			dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
#endif
			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
			dL_dz += dL_dmd * dmd_dd;

			// Propagate gradients w.r.t ray-splat depths
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;
			// Propagate gradients w.r.t. color ray-splat alphas
			accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

			// Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = normal[ch];
				dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
			}
#endif

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * dL_dalpha;
#if RENDER_AXUTILITY
			dL_dz += alpha * T * dL_ddepth; 
#endif

			if (rho3d <= rho2d) {
				// Update gradients w.r.t. covariance of Gaussian 3x3 (T)
				const float2 dL_ds = {
					dL_dG * -G * s.x + dL_dz * Tw.x,
					dL_dG * -G * s.y + dL_dz * Tw.y
				};
				const float3 dz_dTw = {s.x, s.y, 1.0};
				const float dsx_pz = dL_ds.x / p.z;
				const float dsy_pz = dL_ds.y / p.z;
				const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
				const float3 dL_dk = cross(l, dL_dp);
				const float3 dL_dl = cross(dL_dp, k);

				const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
				const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
				const float3 dL_dTw = {
					pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
					pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
					pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};


				// Update gradients w.r.t. 3D covariance (3x3 matrix)
				atomicAdd(&dL_dtransMat[global_id * 9 + 0],  dL_dTu.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 1],  dL_dTu.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 2],  dL_dTu.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 3],  dL_dTv.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 4],  dL_dTv.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 5],  dL_dTv.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 6],  dL_dTw.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7],  dL_dTw.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dTw.z);
			} else {
				// // Update gradients w.r.t. center of Gaussian 2D mean position
				const float dG_ddelx = -G * FilterInvSquare * d.x;
				const float dG_ddely = -G * FilterInvSquare * d.y;
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); // not scaled
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely); // not scaled
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz); // propagate depth loss
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id*4]), G * dL_dalpha);
			atomicAdd(&(dL_dopacity[global_id*4+1]), G * dL_dalpha);
			atomicAdd(&(dL_dopacity[global_id*4+2]), G * dL_dalpha);
			atomicAdd(&(dL_dopacity[global_id*4+3]), G * dL_dalpha);

			// atomicAdd(&(dL_dopacity[global_id*4]),  L3outR.w - my_o.x);
			// atomicAdd(&(dL_dopacity[global_id*4+1]), L3outR.w - my_o.x);
			// atomicAdd(&(dL_dopacity[global_id*4+2]), L3outR.w - my_o.x);
			// atomicAdd(&(dL_dopacity[global_id*4+3]), L3outR.w - my_o.x);
			float dL_dopacity = G * dL_dalpha;





			// atomicAdd(&(dL_dopacity[global_id*4 +0]), G * dL_dalpha *  exp(-1.0f*my_rho[0]*exp_scale[0]) / 4.0f);
			// atomicAdd(&(dL_dopacity[global_id*4 +1]), G * dL_dalpha *  exp(-1.0f*my_rho[1]*exp_scale[1]) / 4.0f);
			// atomicAdd(&(dL_dopacity[global_id*4 +2]), G * dL_dalpha *  exp(-1.0f*my_rho[2]*exp_scale[2]) / 4.0f);
			// atomicAdd(&(dL_dopacity[global_id*4 +3]), G * dL_dalpha *  exp(-1.0f*my_rho[3]*exp_scale[3]) / 4.0f);

			
			glm::vec4 dL_doutput_data = glm::vec4(2.0f*dl_dc[0], 2.0f*dl_dc[1], 2.0f*dl_dc[2], dL_dopacity);
			dL_doutput_data = dL_doutput_data * dL_dSigmoid(L3out);
			glm::vec4 dL_dmiddle_data =  dL_doutput_data* output;

			glm::mat4x4 dL_doutput = glm::mat4x4(
				glm::vec4(L2outR.x*dL_doutput_data.x, L2outR.x*dL_doutput_data.y, L2outR.x*dL_doutput_data.z, L2outR.x*dL_doutput_data.w),
				glm::vec4(L2outR.y*dL_doutput_data.x, L2outR.y*dL_doutput_data.y, L2outR.y*dL_doutput_data.z, L2outR.y*dL_doutput_data.w),
				glm::vec4(L2outR.z*dL_doutput_data.x, L2outR.z*dL_doutput_data.y, L2outR.z*dL_doutput_data.z, L2outR.z*dL_doutput_data.w),
				glm::vec4(L2outR.w*dL_doutput_data.x, L2outR.w*dL_doutput_data.y, L2outR.w*dL_doutput_data.z, L2outR.w*dL_doutput_data.w)
			);
\
			glm::vec4 dL_doutput_biases = dL_doutput_data;

			dL_dmiddle_data= dL_dmiddle_data * dL_dSigmoid(L2out);

			glm::vec4 dL_dinput_data = dL_dmiddle_data * middle;
			// dL_dmiddle = dL_dmiddle_data * L1out;
			glm::mat4x4 dL_dmiddle = glm::mat4x4(
				glm::vec4(L1outR.x*dL_dmiddle_data.x, L1outR.x*dL_dmiddle_data.y, L1outR.x*dL_dmiddle_data.z, L1outR.x*dL_dmiddle_data.w),
				glm::vec4(L1outR.y*dL_dmiddle_data.x, L1outR.y*dL_dmiddle_data.y, L1outR.y*dL_dmiddle_data.z, L1outR.y*dL_dmiddle_data.w),
				glm::vec4(L1outR.z*dL_dmiddle_data.x, L1outR.z*dL_dmiddle_data.y, L1outR.z*dL_dmiddle_data.z, L1outR.z*dL_dmiddle_data.w),
				glm::vec4(L1outR.w*dL_dmiddle_data.x, L1outR.w*dL_dmiddle_data.y, L1outR.w*dL_dmiddle_data.z, L1outR.w*dL_dmiddle_data.w)
			);
			glm::vec4 dL_dmiddle_biases = dL_dmiddle_data;

		
			dL_dinput_data = dL_dinput_data * dL_dSigmoid(L1out);
			glm::vec2 dL_dsxy = dL_dinput_data * input;
			glm::mat2x4 dL_dinput = glm::mat2x4(
				glm::vec4(input_data2.x*dL_dinput_data.x, input_data2.x*dL_dinput_data.y, input_data2.x*dL_dinput_data.z, input_data2.x*dL_dinput_data.w),
				glm::vec4(input_data2.y*dL_dinput_data.x, input_data2.y*dL_dinput_data.y, input_data2.y*dL_dinput_data.z, input_data2.y*dL_dinput_data.w)
			);
			glm::vec4 dL_dinput_biases = dL_dinput_data;

			// dL_dsxy.x = dL_dsxy.x * 1.0 / (1.0 + exp(-s.x)) * (1.0 - 1.0 / (1.0 + exp(-s.x)));
			// dL_dsxy.y = dL_dsxy.y * 1.0 / (1.0 + exp(-s.y)) * (1.0 - 1.0 / (1.0 + exp(-s.y)));

			atomicAdd(&dL_dshs[102*global_id + 48 ], dL_dinput[0].x);
			atomicAdd(&dL_dshs[102*global_id + 49 ], dL_dinput[0].y);
			atomicAdd(&dL_dshs[102*global_id + 50 ], dL_dinput[0].z);
			atomicAdd(&dL_dshs[102*global_id + 51 ], dL_dinput[0].w);
			atomicAdd(&dL_dshs[102*global_id + 52 ], dL_dinput[1].x);
			atomicAdd(&dL_dshs[102*global_id + 53 ], dL_dinput[1].y);
			atomicAdd(&dL_dshs[102*global_id + 54 ], dL_dinput[1].z);
			atomicAdd(&dL_dshs[102*global_id + 55 ], dL_dinput[1].w);
			atomicAdd(&dL_dshs[102*global_id + 56 ], dL_dinput_biases.x);
			atomicAdd(&dL_dshs[102*global_id + 57 ], dL_dinput_biases.y);
			atomicAdd(&dL_dshs[102*global_id + 58 ], dL_dinput_biases.z);
			atomicAdd(&dL_dshs[102*global_id + 59 ], dL_dinput_biases.w);

			atomicAdd(&dL_dshs[102*global_id + 60 ], dL_dmiddle[0].x);
			atomicAdd(&dL_dshs[102*global_id + 61 ], dL_dmiddle[0].y);
			atomicAdd(&dL_dshs[102*global_id + 62 ], dL_dmiddle[0].z);
			atomicAdd(&dL_dshs[102*global_id + 63 ], dL_dmiddle[0].w);
			atomicAdd(&dL_dshs[102*global_id + 64 ], dL_dmiddle[1].x);
			atomicAdd(&dL_dshs[102*global_id + 65 ], dL_dmiddle[1].y);
			atomicAdd(&dL_dshs[102*global_id + 66 ], dL_dmiddle[1].z);
			atomicAdd(&dL_dshs[102*global_id + 67 ], dL_dmiddle[1].w);
			atomicAdd(&dL_dshs[102*global_id + 68 ], dL_dmiddle[2].x);
			atomicAdd(&dL_dshs[102*global_id + 69 ], dL_dmiddle[2].y);
			atomicAdd(&dL_dshs[102*global_id + 70 ], dL_dmiddle[2].z);
			atomicAdd(&dL_dshs[102*global_id + 71 ], dL_dmiddle[2].w);
			atomicAdd(&dL_dshs[102*global_id + 72 ], dL_dmiddle[3].x);
			atomicAdd(&dL_dshs[102*global_id + 73 ], dL_dmiddle[3].y);
			atomicAdd(&dL_dshs[102*global_id + 74 ], dL_dmiddle[3].z);
			atomicAdd(&dL_dshs[102*global_id + 75 ], dL_dmiddle[3].w);
			atomicAdd(&dL_dshs[102*global_id + 76 ], dL_dmiddle_biases.x);
			atomicAdd(&dL_dshs[102*global_id + 77 ], dL_dmiddle_biases.y);
			atomicAdd(&dL_dshs[102*global_id + 78 ], dL_dmiddle_biases.z);
			atomicAdd(&dL_dshs[102*global_id + 79 ], dL_dmiddle_biases.w);

			atomicAdd(&dL_dshs[102*global_id + 80 ], dL_doutput[0].x);
			atomicAdd(&dL_dshs[102*global_id + 81 ], dL_doutput[0].y);
			atomicAdd(&dL_dshs[102*global_id + 82 ], dL_doutput[0].z);
			atomicAdd(&dL_dshs[102*global_id + 83 ], dL_doutput[0].w);
			atomicAdd(&dL_dshs[102*global_id + 84 ], dL_doutput[1].x);
			atomicAdd(&dL_dshs[102*global_id + 85 ], dL_doutput[1].y);
			atomicAdd(&dL_dshs[102*global_id + 86 ], dL_doutput[1].z);
			atomicAdd(&dL_dshs[102*global_id + 87 ], dL_doutput[1].w);
			atomicAdd(&dL_dshs[102*global_id + 88 ], dL_doutput[2].x);
			atomicAdd(&dL_dshs[102*global_id + 89 ], dL_doutput[2].y);
			atomicAdd(&dL_dshs[102*global_id + 90 ], dL_doutput[2].z);
			atomicAdd(&dL_dshs[102*global_id + 91 ], dL_doutput[2].w);
			atomicAdd(&dL_dshs[102*global_id + 92 ], dL_doutput[3].x);
			atomicAdd(&dL_dshs[102*global_id + 93 ], dL_doutput[3].y);
			atomicAdd(&dL_dshs[102*global_id + 94 ], dL_doutput[3].z);
			atomicAdd(&dL_dshs[102*global_id + 95 ], dL_doutput[3].w);
			atomicAdd(&dL_dshs[102*global_id + 96 ], dL_doutput_biases.x);
			atomicAdd(&dL_dshs[102*global_id + 97 ], dL_doutput_biases.y);
			atomicAdd(&dL_dshs[102*global_id + 98 ], dL_doutput_biases.z);
			atomicAdd(&dL_dshs[102*global_id + 99 ], dL_doutput_biases.w);
			



	
				
			// dl_dtu[0] += dL_dsxy[0] * dsx_dtu[0] + dL_dsxy[1] * dsy_dtu[0];
			// dl_dtu[1] += dL_dsxy[0] * dsx_dtu[1] + dL_dsxy[1] * dsy_dtu[1];
			// dl_dtu[2] += dL_dsxy[0] * dsx_dtu[2] + dL_dsxy[1] * dsy_dtu[2];
			// dl_dtv[0] += dL_dsxy[0] * dsx_dtv[0] + dL_dsxy[1] * dsy_dtv[0];
			// dl_dtv[1] += dL_dsxy[0] * dsx_dtv[1] + dL_dsxy[1] * dsy_dtv[1];
			// dl_dtv[2] += dL_dsxy[0] * dsx_dtv[2] + dL_dsxy[1] * dsy_dtv[2];
			// dl_dtw[0] += dL_dsxy[0] * dsx_dtw[0] + dL_dsxy[1] * dsy_dtw[0];
			// dl_dtw[1] += dL_dsxy[0] * dsx_dtw[1] + dL_dsxy[1] * dsy_dtw[1];
			// dl_dtw[2] += dL_dsxy[0] * dsx_dtw[2] + dL_dsxy[1] * dsy_dtw[2];

			// 	atomicAdd(&dL_dtransMat[global_id * 9 + 0], dl_dtu[0]);
			// 	atomicAdd(&dL_dtransMat[global_id * 9 + 1], dl_dtu[1]);
			// 	atomicAdd(&dL_dtransMat[global_id * 9 + 2], dl_dtu[2]);
			// 	atomicAdd(&dL_dtransMat[global_id * 9 + 3], dl_dtv[0]);
			// 	atomicAdd(&dL_dtransMat[global_id * 9 + 4], dl_dtv[1]);
			// 	atomicAdd(&dL_dtransMat[global_id * 9 + 5], dl_dtv[2]);
			// 	atomicAdd(&dL_dtransMat[global_id * 9 + 6], dl_dtw[0]);
			// 	atomicAdd(&dL_dtransMat[global_id * 9 + 7], dl_dtw[1]);
			// 	atomicAdd(&dL_dtransMat[global_id * 9 + 8], dl_dtw[2]);

			
			
				
				// float2 dl_dmeans2d = {0.0,0.0};

				// dc_dsx[0] = Myalpha * (-1.0 * (1.0 - proj_v) * addrgb[0].x - (proj_v) * addrgb[1].x + (1.0-proj_v) * addrgb[2].x + proj_v * addrgb[3].x)   * ((sigmod_scale*exp(-sigmod_scale*d.x))/((exp(-sigmod_scale*d.x) + 1.0)*(exp(-sigmod_scale*d.x) + 1.0)));
				// dc_dsx[1] = Myalpha * (-1.0 * (1.0 - proj_v) * addrgb[0].y - (proj_v) * addrgb[1].y + (1.0-proj_v) * addrgb[2].y + proj_v * addrgb[3].y)   * ((sigmod_scale*exp(-sigmod_scale*d.x))/((exp(-sigmod_scale*d.x) + 1.0)*(exp(-sigmod_scale*d.x) + 1.0)));
				// dc_dsx[2] = Myalpha * (-1.0 * (1.0 - proj_v) * addrgb[0].z - (proj_v) * addrgb[1].z + (1.0-proj_v) * addrgb[2].z + proj_v * addrgb[3].z)   * ((sigmod_scale*exp(-sigmod_scale*d.x))/((exp(-sigmod_scale*d.x) + 1.0)*(exp(-sigmod_scale*d.x) + 1.0)));
				// dc_dsy[0] = Myalpha * (-1.0 * (1.0 - proj_u) * addrgb[0].x + (1.0 - proj_u) * addrgb[1].x - (proj_u) * addrgb[2].x + proj_u * addrgb[3].x) * ((sigmod_scale*exp(-sigmod_scale*d.y))/((exp(-sigmod_scale*d.y) + 1.0)*(exp(-sigmod_scale*d.y) + 1.0)));
				// dc_dsy[1] = Myalpha * (-1.0 * (1.0 - proj_u) * addrgb[0].y + (1.0 - proj_u) * addrgb[1].y - (proj_u) * addrgb[2].y + proj_u * addrgb[3].y) * ((sigmod_scale*exp(-sigmod_scale*d.y))/((exp(-sigmod_scale*d.y) + 1.0)*(exp(-sigmod_scale*d.y) + 1.0)));
				// dc_dsy[2] = Myalpha * (-1.0 * (1.0 - proj_u) * addrgb[0].z + (1.0 - proj_u) * addrgb[1].z - (proj_u) * addrgb[2].z + proj_u * addrgb[3].z) * ((sigmod_scale*exp(-sigmod_scale*d.y))/((exp(-sigmod_scale*d.y) + 1.0)*(exp(-sigmod_scale*d.y) + 1.0)));

				// dop_dsx = (-1.0 * (1.0 - proj_v) * my_o.x - (proj_v) * my_o.y + (1.0-proj_v) * my_o.z + proj_v * my_o.w)   * ((sigmod_scale*exp(-sigmod_scale*d.x))/((exp(-sigmod_scale*d.x) + 1.0)*(exp(-sigmod_scale*d.x) + 1.0)));
				// dop_dsy = (-1.0 * (1.0 - proj_u) * my_o.x + (1.0 - proj_u) * my_o.y - (proj_u) * my_o.z + proj_u * my_o.w) * ((sigmod_scale*exp(-sigmod_scale*d.y))/((exp(-sigmod_scale*d.y) + 1.0)*(exp(-sigmod_scale*d.y) + 1.0)));
				// for (int ch = 0; ch < C; ch++)
				// {
				// 	dl_dmeans2d.x += dl_dc[ch] * dc_dsx[ch];
				// 	dl_dmeans2d.y += dl_dc[ch] * dc_dsy[ch];
				// }
				// dl_dmeans2d.x += G * dL_dalpha * dop_dsx;
				// dl_dmeans2d.y += G * dL_dalpha * dop_dsy;

				// atomicAdd(&dL_dmean2D[global_id].x, dl_dmeans2d.x);
				// atomicAdd(&dL_dmean2D[global_id].y, dl_dmeans2d.y); 
				
				
			

		


		}
	}
}


__device__ void compute_transmat_aabb(
	int idx, 
	const float* Ts_precomp,
	const float3* p_origs, 
	const glm::vec2* scales, 
	const glm::vec4* rots, 
	const float* projmatrix, 
	const float* viewmatrix, 
	const int W, const int H, 
	const float3* dL_dnormals,
	const float3* dL_dmean2Ds, 
	float* dL_dTs, 
	glm::vec3* dL_dmeans, 
	glm::vec2* dL_dscales,
	 glm::vec4* dL_drots)
{
	glm::mat3 T;
	float3 normal;
	glm::mat3x4 P;
	glm::mat3 R;
	glm::mat3 S;
	float3 p_orig;
	glm::vec4 rot;
	glm::vec2 scale;
	
	// Get transformation matrix of the Gaussian
	if (Ts_precomp != nullptr) {
		T = glm::mat3(
			Ts_precomp[idx * 9 + 0], Ts_precomp[idx * 9 + 1], Ts_precomp[idx * 9 + 2],
			Ts_precomp[idx * 9 + 3], Ts_precomp[idx * 9 + 4], Ts_precomp[idx * 9 + 5],
			Ts_precomp[idx * 9 + 6], Ts_precomp[idx * 9 + 7], Ts_precomp[idx * 9 + 8]
		);
		normal = {0.0, 0.0, 0.0};
	} else {
		p_orig = p_origs[idx];
		rot = rots[idx];
		scale = scales[idx];
		R = quat_to_rotmat(rot);
		S = scale_to_mat(scale, 1.0f);
		
		glm::mat3 L = R * S;
		glm::mat3x4 M = glm::mat3x4(
			glm::vec4(L[0], 0.0),
			glm::vec4(L[1], 0.0),
			glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
		);

		glm::mat4 world2ndc = glm::mat4(
			projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
			projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
			projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
			projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
		);

		glm::mat3x4 ndc2pix = glm::mat3x4(
			glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
			glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
			glm::vec4(0.0, 0.0, 0.0, 1.0)
		);

		P = world2ndc * ndc2pix;
		T = glm::transpose(M) * P;
		normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
	}

	// Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx*9+0], dL_dTs[idx*9+1], dL_dTs[idx*9+2],
		dL_dTs[idx*9+3], dL_dTs[idx*9+4], dL_dTs[idx*9+5],
		dL_dTs[idx*9+6], dL_dTs[idx*9+7], dL_dTs[idx*9+8]
	);
	float3 dL_dmean2D = dL_dmean2Ds[idx];
	if(dL_dmean2D.x != 0 || dL_dmean2D.y != 0)
	{
		glm::vec3 t_vec = glm::vec3(9.0f, 9.0f, -1.0f);
		float d = glm::dot(t_vec, T[2] * T[2]);
		glm::vec3 f_vec = t_vec * (1.0f / d);
		glm::vec3 dL_dT0 = dL_dmean2D.x * f_vec * T[2];
		glm::vec3 dL_dT1 = dL_dmean2D.y * f_vec * T[2];
		glm::vec3 dL_dT3 = dL_dmean2D.x * f_vec * T[0] + dL_dmean2D.y * f_vec * T[1];
		glm::vec3 dL_df = dL_dmean2D.x * T[0] * T[2] + dL_dmean2D.y * T[1] * T[2];
		float dL_dd = glm::dot(dL_df, f_vec) * (-1.0 / d);
		glm::vec3 dd_dT3 = t_vec * T[2] * 2.0f;
		dL_dT3 += dL_dd * dd_dT3;
		dL_dT[0] += dL_dT0;
		dL_dT[1] += dL_dT1;
		dL_dT[2] += dL_dT3;

		if (Ts_precomp != nullptr) {
			dL_dTs[idx * 9 + 0] = dL_dT[0].x;
			dL_dTs[idx * 9 + 1] = dL_dT[0].y;
			dL_dTs[idx * 9 + 2] = dL_dT[0].z;
			dL_dTs[idx * 9 + 3] = dL_dT[1].x;
			dL_dTs[idx * 9 + 4] = dL_dT[1].y;
			dL_dTs[idx * 9 + 5] = dL_dT[1].z;
			dL_dTs[idx * 9 + 6] = dL_dT[2].x;
			dL_dTs[idx * 9 + 7] = dL_dT[2].y;
			dL_dTs[idx * 9 + 8] = dL_dT[2].z;
			return;
		}
	}
	
	if (Ts_precomp != nullptr) return;

	// Update gradients w.r.t. scaling, rotation, position of the Gaussian
	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormals[idx], viewmatrix);
#if DUAL_VISIABLE
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float cos = -sumf3(p_view * normal);
	float multiplier = cos > 0 ? 1: -1;
	dL_dtn = multiplier * dL_dtn;
#endif
	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dM[0]),
		glm::vec3(dL_dM[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
	);

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS[0] * glm::vec3(scale.x),
		dL_dRS[1] * glm::vec3(scale.y),
		dL_dRS[2]);
	
	dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR);
	dL_dscales[idx] = glm::vec2(
		(float)glm::dot(dL_dRS[0], R[0]),
		(float)glm::dot(dL_dRS[1], R[1])
	);
	dL_dmeans[idx] = glm::vec3(dL_dM[2]);
}

template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means3D,
	const float* transMats,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, 
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* campos, 
	// grad input
	float* dL_dtransMats,
	const float* dL_dnormal3Ds,
	float* dL_dcolors,
	float* dL_dshs,
	float3* dL_dmean2Ds,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots,
	float2* wichzone)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int W = int(focal_x * tan_fovx * 2);
	const int H = int(focal_y * tan_fovy * 2);
	const float * Ts_precomp = (scales) ? nullptr : transMats;
	compute_transmat_aabb(
		idx, 
		Ts_precomp,
		means3D, scales, rotations, 
		projmatrix, viewmatrix, W, H, 
		(float3*)dL_dnormal3Ds, 
		dL_dmean2Ds,
		(dL_dtransMats), 
		dL_dmean3Ds, 
		dL_dscales, 
		dL_drots
	);

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *campos, shs, clamped, (glm::vec3*)dL_dcolors, (glm::vec3*)dL_dmean3Ds, (glm::vec3*)dL_dshs, wichzone);
	
	// hack the gradient here for densitification
	float depth = transMats[idx * 9 + 8];
	dL_dmean2Ds[idx].x = dL_dtransMats[idx * 9 + 2] * depth * 0.5 * float(W); // to ndc 
	dL_dmean2Ds[idx].y = dL_dtransMats[idx * 9 + 5] * depth * 0.5 * float(H); // to ndc
}


void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* transMats,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	const glm::vec3* campos, 
	float3* dL_dmean2Ds,
	const float* dL_dnormal3Ds,
	float* dL_dtransMats,
	float* dL_dcolors,
	float* dL_dshs,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots,
	float2* wichzone)
{	
	preprocessCUDA<NUM_CHANNELS><< <(P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		transMats,
		radii,
		shs,
		clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		focal_x, 
		focal_y,
		tan_fovx,
		tan_fovy,
		campos,	
		dL_dtransMats,
		dL_dnormal3Ds,
		dL_dcolors,
		dL_dshs,
		dL_dmean2Ds,
		dL_dmean3Ds,
		dL_dscales,
		dL_drots,
		wichzone
	);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* bg_color,
	const float2* means2D,
	const float4* normal_opacity,
	const float4* my_opacity,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_depths,
	float * dL_dtransMat,
	float3* dL_dmean2D,
	float* dL_dnormal3D,
	float* dL_dopacity,
	float* dL_dcolors,
	const float* shs,
	float2* wichzone,
	float* dL_dshs)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		bg_color,
		means2D,
		normal_opacity,
		my_opacity,
		transMats,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_depths,
		dL_dtransMat,
		dL_dmean2D,
		dL_dnormal3D,
		dL_dopacity,
		dL_dcolors,
		shs,
		wichzone,
		dL_dshs
		);
}
