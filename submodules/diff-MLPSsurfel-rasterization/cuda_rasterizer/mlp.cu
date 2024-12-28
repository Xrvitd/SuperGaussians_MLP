#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <assert.h>
#include <cublas_v2.h>
#include <curand.h>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

__global__ void bias_addition(int n, double *A, double *B, double *C, int sign = 1) { // change sign for subtraction or scaled addition
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n)
			return;
		C[index] = A[index] + sign*B[index];
	}

__global__ void relu_activation(int n, double *A, double *C) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n)
		return;
	C[index] = max(0.0f, A[index]);
}

__global__ void relu_grad(int n, double *z, double * grad) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n)
		return;
	grad[index] = (z[index] > 0) ? 1 : 0;; // makes a diagona matrix
}
__global__ void softmax_activation(int n, double *A, double *C, double exp_sum) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n)
		return;
	C[index] = exp(A[index]) / exp_sum;
}
__global__ void scan(int n, double *data, int d) {// function to get sum (for softmax layer)
	int tmp_d = 1 << (d + 1);
	int index = (blockDim.x * blockIdx.x + threadIdx.x)*tmp_d;
	if (index >= n)
		return;
	data[index + tmp_d - 1] += data[index + (tmp_d >> 1) - 1];
}
__global__ void exp_copy(int n, double *odata, double *idata) {// kernal to copy exp(idata[i]) to odata[i]
	int index = (blockDim.x * blockIdx.x + threadIdx.x);
	if (index >= n)
		return;
	odata[index] = exp(idata[index]);
}
__global__ void fill_data(int n, double *data, double val) {
	int index = (blockDim.x * blockIdx.x + threadIdx.x);
	if (index >= n)
		return;
	data[index] = val;
}
__global__ void element_mult(int n, double *a, double *b, double *c) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n)
		return;
	c[index] = a[index] * b[index];
}
__global__ void update_params(int n, double *param, double *grad, double lr) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n)
		return;
	param[index] -= lr * grad[index];
}
__global__ void memset(int n, double *data, float value) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n)
		return;
	data[index] = value;
}
__global__ void update_momentum(int n, double *vdw, double *dL_dw, double beta) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n)
		return;
	vdw[index] = beta * vdw[index] + (1 - beta) * dL_dw[index];
}
__global__ void cross_entropy_kernal(int n, double *y, double *y_hat, double *dev_loss) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n)
		return;
	dev_loss[index] = y[index] * log2(y_hat[index]);
}
/*From Stream compaction part of this assignment*/
__global__ void reduce_kern(int n, double *data, int d) {
	int tmp_d = 1 << (d + 1);
	int index = (blockDim.x * blockIdx.x + threadIdx.x)*tmp_d;
	if (index >= n)
		return;
	double tmp_data = data[index + (tmp_d >> 1) - 1]; // saves a read or write
	if (tmp_data == 0)
		return;
	data[index + tmp_d - 1] += tmp_data;
}










Net::Net(int n, vector<int> layers, double lr, double beta) {
		// layers = {98, 52, 52}
		params.layer_count = layers.size();
		params.input_size = n;
		params.output_size = layers[params.layer_count - 1];
		params.lr = lr;
		params.layer_sizes = layers;
		if (beta != -1) {
			momentum_grad = true;
			params.beta = beta;
		}
		else
			momentum_grad = false;
		// init raw data holder
		cudaMalloc((void**)&dev_data, n * sizeof(double));
		cudaMalloc((void**)&dev_y, params.output_size * sizeof(double));
		cudaMalloc((void**)&dev_reduction_pow2, 1<<(ilog2ceil(params.output_size)) * sizeof(double));
		// add input layer to front
		layers.insert(layers.begin(), n);
		double *dev_w, *dev_b, *dev_z, *dev_a, *dev_da;
		int blocks;
		for (int i = 0; i < params.layer_count; i++) {
			cudaMalloc((void**)&dev_w, (layers[i] * layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			cudaMalloc((void**)&dev_b, (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			// initilize w, b using gaussian distribution
			GPU_fill_rand(dev_w, layers[i] * layers[i + 1], 2.0 / (layers[i])); // uniform random initilization
			// memset dev_b
			blocks = (layers[i + 1] + params.block_size - 1) / params.block_size;
			memset << <blocks, params.block_size >> > (layers[i + 1], dev_b, 0.1);
			checkCUDAErrorWithLine("Memset failed!");
			//GPU_fill_rand(dev_b, layers[i + 1], 2.0 / layers[i]); // zero initilizaton is fine for biases
			// push into vector
			w.push_back(dev_w);
			b.push_back(dev_b);
			// intermediate results arrays
			cudaMalloc((void**)&dev_z, (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			cudaMalloc((void**)&dev_a, (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			z.push_back(dev_z);
			a.push_back(dev_a);

			// grad arrays
			cudaMalloc((void**)&dev_w, (layers[i] * layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			dL_dw.push_back(dev_w); // gradient of w

			cudaMalloc((void**)&dev_da, (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			dL_dz.push_back(dev_da); // da/dg has dimensions output(g) * output(g) <Jacobian>

			cudaMalloc((void**)&dev_da, (layers[i]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			dL_da.push_back(dev_da); // da/dg has dimensions output(g) * output(g) <Jacobian>
			// relu grad
			cudaMalloc((void**)&dev_z, (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			d_relu.push_back(dev_z);
			// momentum variables
			if (momentum_grad) {
				// Vb
				cudaMalloc((void**)&dev_z, (layers[i + 1]) * sizeof(double));
				checkCUDAErrorWithLine("Malloc failed!");
				memset << <blocks, params.block_size >> > (layers[i + 1], dev_z, 0);// zero position because it is a running buffer
				checkCUDAErrorWithLine("Memset failed!");
				vdb.push_back(dev_z);
				// Vw
				blocks = ((layers[i] * layers[i + 1]) + params.block_size - 1) / params.block_size;
				cudaMalloc((void**)&dev_w, (layers[i] * layers[i + 1]) * sizeof(double));
				memset << <blocks, params.block_size >> > ((layers[i] * layers[i + 1]), dev_w, 0); // zero position because it is a running buffer
				checkCUDAErrorWithLine("Cuda malloc failed!");
				vdw.push_back(dev_w);
			}
		}
		// initilizaton cublas handle
		cublasCreate(&handle); 
		// set read_dev_y flag to false (i.e not read data)
		read_dev_y = false;
	}



