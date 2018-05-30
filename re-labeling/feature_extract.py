import argparse
import matplotlib.pyplot as plot

import chainer
import cupy

from PIL import Image
import numpy as np
import pandas as pd

import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--root_path', default="/data/unagi0/food/train_set/")
    parser.add_argument('--image_list', default="/data/unagi0/food/tmp/train_full.txt")
    parser.add_argument('--dst_path', default="/data/unagi0/food/tmp/dst_feature/")
    args = parser.parse_args()

    image_list = open(args.image_list, "r")
    feature_array = np.zeros((101733, 4096), dtype=np.float32)

    model = chainer.links.VGG16Layers()
    if args.gpu > 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    for i, filename in enumerate(image_list):
        filename = filename.split('\n')[0]
        print(filename)

        img = Image.open(args.root_path + filename)
        feature = model.extract([img], layers=['fc6'])
        feature_array[i] = chainer.cuda.to_cpu(feature['fc6'][0].data)

    np.save(args.dst_path + "full", feature_array)

if __name__ == "__main__":
    main()
