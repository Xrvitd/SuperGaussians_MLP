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
nerf_scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"] 

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="/data6/ruixu/Eval/ALL/1030_MLP_FINAL_video")
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
    parser.add_argument("--tanksandtemples", "-tat", default="/data6/ruixu/data/tandt")
    parser.add_argument("--deepblending", "-db", default="/data6/ruixu/data/db")
    args = parser.parse_args()

if not args.skip_training:
    common_args = " --quiet --eval"
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/tat/" + scene + common_args)
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/db/" + scene + common_args)
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/m360/" + scene + common_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/m360/" + scene + common_args)

    common_args = " --eval --quiet"
    for scene in nerf_scenes:
        source = args.nerf + "/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/nerf/" + scene + common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/nerf/" + scene + common_args)
    common_args = " -r 2"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/DTU/" + scene + common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/DTU/" + scene + common_args)

    
if not args.skip_rendering:
    common_args = " --quiet --eval --skip_train --render_path"


    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/tat/" + scene + common_args)
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/db/" + scene + common_args)
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python render.py --iteration 30000 -s " + source + " -i images_4 -m " + args.output_path + "/m360/" + scene + common_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python render.py --iteration 30000 -s " + source + " -i images_2 -m " + args.output_path + "/m360/" + scene + common_args)

    # common_args = " --eval --quiet"
    for scene in nerf_scenes:
        source = args.nerf + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/nerf/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/nerf/" + scene + common_args)
    # common_args = " -r 2"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/DTU/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/DTU/" + scene + common_args)
