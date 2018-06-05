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

import finetune
import pretrainedmodels

# import torch
# import torch.multiprocessing
# torch.multiprocessing.set_start_method('spawn')

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='ci', type=str,
                    help='dataset (ifood18[default] or cifar10 or cifar100)')
parser.add_argument('--epochs', default=400, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=4, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.5, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')

parser.add_argument('--validate_freq', '-valf', default=2, type=int,
                    help='validate frequency (default: 100)')
parser.add_argument('--BC', help='Use BC learning', action='store_true')
parser.add_argument('--BCp', help='Use BC learning', action='store_true')
parser.add_argument('--model', default="resnet152", help='Use BC learning')

parser.set_defaults(augment=True)

best_prec3 = 0


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
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # translate by -20 to +20 percent (per axis)
                rotate=(-20, 20),  # rotate by -45 to +45 degrees
                shear=(-10, 10),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 2),
                       [
                           # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 2.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 4)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(3, 5)),
                               # blur image using local medians with kernel sizes between 2 and 7
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
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
                           # iaa.Invert(0.05, per_channel=True),  # invert color channels
                           iaa.Add((-10, 10), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           # iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           iaa.OneOf([
                               iaa.Multiply((0.9, 1.1), per_channel=0.5),
                               # iaa.FrequencyNoiseAlpha(
                               #     exponent=(-4, 0),
                               #     first=iaa.Multiply((0.5, 1.5), per_channel=True),
                               #     second=iaa.ContrastNormalization((0.5, 2.0))
                               # )
                           ]),
                           iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5),  # improve or worsen the contrast
                           # iaa.Grayscale(alpha=(0.0, 1.0)),
                           # sometimes(iaa.ElasticTransformation(alpha=(0.5, 1.0), sigma=0.25)),
                           # move pixels locally around (with random strengths)
                           # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           # sometimes move parts of the image around
                           # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )

    # print("type ", type(image))

    return seq.augment_image(image)


class FoodDataset(Dataset):
    """Food dataset."""

    def __init__(self, root_dir, csv_file, transform, training=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.labels = pd.read_csv(csv_file)

        data = pd.read_csv(csv_file, header=None, names=["name_of_pic", "noisy_label"])
        # print(selfdata.head())
        self.labels = data.noisy_label.tolist()
        # read addresses and labels from the 'train' folder
        self.pic_names = data.name_of_pic.tolist()
        # print("Length of val data is : ",len(val_data))

        self.root_dir = root_dir
        self.flag = training
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.pic_names[idx])

        image = ndimage.imread(img_name, mode="RGB")
        image = misc.imresize(image, (256, 256), mode='RGB')
        # image = transform.resize(image , (128,128))
        #
        # image = resizeimage.resize_cover(Image.open(img_name), [128, 128])

        if (self.flag == True):
            image = augment_images(image)

        correct_label = self.labels[idx]
        # correct_label = correct_label.astype('int')
        if self.transform:
            image = self.transform(image)

        return (image, correct_label)


def main():
    global args, best_prec3
    args = parser.parse_args()
    assert args.model in ["resnet152", "se_resnext101_32x4d"]

    # Data loading code
    # normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
    #								 std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Lambda(lambda x: F.pad(
            #					Variable(x.unsqueeze(0), requires_grad=False, volatile=True),
            #					(4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            # transforms.Resize(192,192),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            # normalize,
        ])
    else:
        transform_train = transforms.Compose([
            # transforms.Resize(256,256),
            transforms.ToTensor()
            # normalize,
        ])

    val_data_path = "/home/mil/gupta/ifood18/data/val_set/"

    val_label = "/home/mil/gupta/ifood18/data/labels/val_info.csv"

    val_dataset = FoodDataset(val_data_path, val_label, transform_train, training=True)

    batchsize = 768

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batchsize, num_workers=16)

    # model = pretrainedmodels.__dict__[args.model](num_classes=1000, pretrained='imagenet')
    # model = finetune.FineTuneModel(model)

    # get the number of model parameters
    # print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    # optionally resume from a checkpoint
    models = []
    for resume in args.resume.split(" + "):
        if resume:
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                args.start_epoch = checkpoint['epoch']
                best_prec3 = checkpoint['best_prec3']
                models.append(checkpoint['model'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(resume))
    assert models, "no model matched"
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.KLDivLoss(size_average=False).cuda()
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
    #                             momentum=args.momentum,  # nesterov=args.nesterov,
    #                             weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #                              args.lr,
    #                              weight_decay=args.weight_decay)

    target = torch.from_numpy(np.array(val_dataset.labels))
    eye = torch.eye(211)
    target = eye[target].cuda()

    output = 0
    for epoch in range(args.start_epoch, args.epochs):
        for model in models:
            prec3, out = validate(val_loader, model, criterion, epoch)
            output += out
        print("aug accuracy",epoch,  accuracy(output, target, (1, 3)))

        best_prec3 = max(prec3, best_prec3)

    print('Best accuracy: ', best_prec3)


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    outputs = []
    for i, (input, target) in enumerate(val_loader):
        batchsize = input.shape[0]

        target = target.type(torch.LongTensor)
        eye = torch.eye(211)
        target = eye[target]

        if args.BCp:
            # subtract mean from image
            mean = torch.mean(input.view(input.shape[0], -1), dim=1)
            input = input - mean.view(-1, 1, 1, 1)

        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            outputs.append(output.data)
        loss = criterion(F.log_softmax(output, 1), target_var) / batchsize

        # measure accuracy and record loss
        prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], input.size(0))
        top3.update(prec3, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top3=top3))

    print(' * Prec@3 {top3.avg:.3f}'.format(top3=top3))
    return top3.avg, torch.cat(outputs)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        # def debug(self):
        # 	print("val = ",self.val)
        # 	print("sum = ",self.sum)
        # 	print("count = ",self.count)
        # 	print("avg = ",self.avg)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    target = torch.argmax(target, dim=1)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:maxk].view(-1).float().sum(0)
    res = (correct_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
