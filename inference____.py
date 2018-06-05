from __future__ import print_function, division
import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from tqdm import tqdm
import cv2, sys
from PIL import Image

from scipy import ndimage, misc
from skimage import data, transform

import matplotlib.pyplot as plt
import six.moves as sm
import re
import os
from collections import defaultdict
import PIL.Image

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

from resizeimage import resizeimage

from scipy import ndimage
import argparse
import os
import shutil
import time
import h5py
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable

from skimage import io, transform

from wideresnet import WideResNet

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value
# import torch
# import torch.multiprocessing
# torch.multiprocessing.set_start_method('spawn')

import glob

import datetime

import finetune
import pretrainedmodels

parser = argparse.ArgumentParser(description='PyTorch ifood18 inferencing')
parser.add_argument('--save-file', default='/home/mil/gupta/ifood18/runs/', type=str,
                    help='location of saved pt model file to load weights from')
parser.add_argument('--test-data', default='/home/mil/gupta/ifood18/data/test_set/', type=str,
                    help='location of images to run inference on')

parser.add_argument('--output-text', default='/home/mil/gupta/ifood18/data/result_test.txt', type=str,
                    help='location of images to run inference on')

parser.add_argument('--name-save-file', default='WideResNet-ifood-28-4-otf-adam-with-augmentation/', type=str,
                    help='name of run')

parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=4, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.5, type=float,
                    help='dropout probability (default: 0.0)')

args = parser.parse_args()


## load
# model = Model() # the model should be defined with the same code you used to create the trained model




def augment_images(image):
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            # sometimes(iaa.CropAndPad(
            #	percent=(-0.05, 0.1),
            #	pad_mode=ia.ALL,
            #	pad_cval=(0, 255)
            # )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(3, 11)),
                               # blur image using local medians with kernel sizes between 2 and 7
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                           # search either for all edges or for directed edges,
                           # blend the result with the original image using a blobby mask
                           # iaa.SimplexNoiseAlpha(iaa.OneOf([
                           #	iaa.EdgeDetect(alpha=(0.5, 1.0)),
                           #	iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           # ])),
                           # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                           # iaa.OneOf([
                           #	iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                           #	iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           # ]),
                           iaa.Invert(0.05, per_channel=True),  # invert color channels
                           iaa.Add((-10, 10), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           iaa.OneOf([
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               iaa.FrequencyNoiseAlpha(
                                   exponent=(-4, 0),
                                   first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                   second=iaa.ContrastNormalization((0.5, 2.0))
                               )
                           ]),
                           iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           iaa.Grayscale(alpha=(0.0, 1.0)),
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                           # move pixels locally around (with random strengths)
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           # sometimes move parts of the image around
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )

    # print("type ", type(image))

    return seq.augment_image(image)


class FoodDatasetTest(Dataset):
    """Food dataset."""

    def __init__(self, root_dir, csv_file, transform=None, training=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.labels = pd.read_csv(csv_file)

        data = pd.read_csv(csv_file, header=None, names=["name_of_pic"])
        # print(selfdata.head())
        # read addresses and labels from the 'train' folder
        self.pic_names = data.name_of_pic.tolist()
        # print("Length of val data is : ",len(val_data))

        self.root_dir = root_dir
        self.flag = training
        self.transform = transform

    def __len__(self):
        return len(self.pic_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.pic_names[idx])
        name = self.pic_names[idx]

        image = ndimage.imread(img_name, mode="RGB")
        image = misc.imresize(image, (256, 256), mode='RGB')
        # image = transform.resize(image , (128,128))
        #
        # image = resizeimage.resize_cover(Image.open(img_name), [128, 128])

        if (self.flag == True):
            image = augment_images(image)

        if self.transform:
            image = self.transform(image)

        return (image, name)


# transform_test = transforms.Compose([transforms.ToPILImage(),
#                                      # transforms.Resize(192,192),
#                                      transforms.CenterCrop(224),
#                                      # transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor()
#                                      # normalize
#                                      ])
transform_test = transforms.Compose([transforms.ToPILImage(),
                                     transforms.RandomCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()
                                     # normalize,
                                     ])

# create model
# model = WideResNet(args.layers, 211, args.widen_factor, dropRate=args.droprate)
model = pretrainedmodels.__dict__["resnet152"](num_classes=1000, pretrained='imagenet')
model = finetune.FineTuneModel(model)

# get the number of model parameters
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# for training on multiple GPUs.
# Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
model = torch.nn.DataParallel(model).cuda()

model_path = os.path.join(args.save_file, args.name_save_file)
print("loading model from ", model_path)
state_dict = torch.load(model_path + "model_best.pth.tar")
checkpoint = torch.load(model_path + "checkpoint.pth.tar")

model.load_state_dict(state_dict['state_dict'])

now = datetime.datetime.now()


def test(test_loader, model):
    """Perform test on the test set"""

    global args
    # switch to evaluate mode
    model.eval()

    end = time.time()

    with open(args.output_text, "w") as file:

        file.write("id,predicted\n")

        for i, (inp, name) in tqdm(enumerate(test_loader)):

            # target = target.type(torch.LongTensor).cuda(async=True)
            # inp = inp.cuda()
            input_var = torch.autograd.Variable(inp, volatile=True)
            # target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            _, pred = output.data.topk(3, 1, True, True)

            # test_000000.jpg,127 121 99
            # loss = criterion(output, target_var)
            for row in range(len(pred)):
                file.write("%s,%d %d %d\n" % (name[row], pred[row][0], pred[row][1], pred[row][2]))
                # measure accuracy and record loss
                # prec3 = accuracy(output.data, target, topk=(1,3))
                # losses.update(loss.data[0], inp.size(0))
                # top3.update(prec3, inp.size(0))

                # measure elapsed time
                # batch_time.update(time.time() - end)

    print("Completed testing and saved the corresponsding file")
    print("Took time : ", time.time() - end)
    return 0


test_data_path = "/home/mil/gupta/ifood18/data/test_set/"

test_csv = "/home/mil/gupta/ifood18/data/labels/test_info.csv"

test_dataset = FoodDatasetTest(test_data_path, test_csv, transform=transform_test)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=4)

# train for one epoch
temp = test(test_loader, model)
