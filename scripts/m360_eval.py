#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser

mipnerf360_outdoor_scenes =  ["bicycle", "flowers", "garden", "stump", "treehill"] #
mipnerf360_indoor_scenes =["room", "counter", "kitchen","bonsai"] #
tanks_and_temples_scenes = ["truck", "train"]
# deep_blending_scenes = ["drjohnson", "playroom"]
# 
# tnt_360_scenes = ['Barn', 'Caterpillar', 'Ignatius', 'Truck']
# tnt_large_scenes = ['Meetingroom', 'Courthouse']
 

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="/data6/ruixu/Eval/m360/1029_MLP_FINAL")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
# all_scenes.extend(mipnerf360_indoor_scenes)
# all_scenes.extend(tanks_and_temples_scenes)
# all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", default="/data6/ruixu/data/360_v2")
    parser.add_argument('--nerf', "-nerf", default="/data6/ruixu/data/nerf_synthetic")
    parser.add_argument('--dtu', "-dtu", default="/data6/ruixu/data/DTU")
    parser.add_argument('--TNT_GT', default="/data6/ruixu/data/tandt")
    # parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    # parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:
    common_args = " --quiet --eval"
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python train2.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args)
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

    # common_args = " --quiet --depth_ratio 1.0 --eval"
    
    # for scene in tnt_360_scenes:
    #     source = args.TNT_data + "/" + scene
    #     print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + ' --lambda_dist 100')
    #     os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

    # for scene in tnt_large_scenes:
    #     source = args.TNT_data + "/" + scene
    #     print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args+ ' --lambda_dist 10')
    #     os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    # for scene in deep_blending_scenes:
        # source = args.deepblending + "/" + scene
        # os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

# if not args.skip_rendering:
#     all_sources = []
#     for scene in mipnerf360_outdoor_scenes:
#         all_sources.append(args.mipnerf360 + "/" + scene)
#     # for scene in mipnerf360_indoor_scenes:
#     #     all_sources.append(args.mipnerf360 + "/" + scene)
#     # for scene in tanks_and_temples_scenes:
#         # all_sources.append(args.tanksandtemples + "/" + scene)
#     # for scene in deep_blending_scenes:
#         # all_sources.append(args.deepblending + "/" + scene)

#     common_args = " --quiet --eval --skip_train"
#     for scene, source in zip(all_scenes, all_sources):
#         os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

# if not args.skip_metrics:
#     scenes_string = ""
#     for scene in all_scenes:
#         scenes_string += "\"" + args.output_path + "/" + scene + "\" "
    
#     os.system("python metrics.py -m " + scenes_string)