import os
from argparse import ArgumentParser

dtu_scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")

parser.add_argument("--output_path", default="/data6/ruixu/code/Eval/NeRF/MLP_nolimit")
parser.add_argument('--nerf', "-nerf", required=True, type=str)
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(dtu_scenes)



if not args.skip_training:
    common_args = " --eval --quiet"
    for scene in dtu_scenes:
        source = args.nerf + "/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

