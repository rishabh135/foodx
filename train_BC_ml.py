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
parser.add_argument('--dataset', default='ifood', type=str,
                    help='dataset (ifood[default] or food101N')
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
parser.add_argument('--name', type=str,
                    help='name of experiment', required=True)
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')

parser.add_argument('--validate_freq', '-valf', default=2, type=int,
                    help='validate frequency (default: 100)')
parser.add_argument('--BC', help='Use BC learning', action='store_true')
parser.add_argument('--BCp', help='Use BC+ learning', action='store_true')
parser.add_argument('--class_weight', help='Use class weight', action='store_true')
parser.add_argument('--model', default="resnet152", help='Use BC learning')
parser.add_argument('--debug', help='debug mode', action='store_true')

parser.set_defaults(augment=True)

best_prec3 = 0


class augment_images:
    def __init__(self):
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.seq = iaa.Sequential(
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
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
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

    def __call__(self, image):
        return self.seq.augment_image(image)


# print(type(tmp))
# grid = seq.draw_grid(image, cols=8, rows=8)
# misc.imsave("examples_grid.jpg", grid)
# io.imsave("examples_image.jpg", tmp)



# return tmp





class FoodDataset(Dataset):
    """Food dataset."""

    def __init__(self, mode, images, transform, training=False):
        self.flag = training
        self.transform = transform
        self.augment_images = augment_images()
        if os.path.exists(f"/data/ugui0/noguchi/ifood/{mode}_images.npy"):
            self.images = images

            self.labels = np.load(f"/data/ugui0/noguchi/ifood/{mode}_labels.npy")
            self.label_weight = np.load(f"/data/ugui0/noguchi/ifood/{mode}_label_weight.npy")
            # read addresses and labels from the 'train' folder
            self.pic_names = np.load(f"/data/ugui0/noguchi/ifood/{mode}_pic_names.npy")
        else:
            csv_file = f"/home/mil/gupta/ifood18/data/labels/{mode}_info.csv"
            data = pd.read_csv(csv_file, header=None, names=["name_of_pic", "noisy_label"])

            self.labels = data.noisy_label.tolist()
            num_label = np.bincount(self.labels)
            self.label_weight = 1 / num_label
            self.label_weight = self.label_weight / np.mean(self.label_weight)
            # read addresses and labels from the 'train' folder
            self.pic_names = data.name_of_pic.tolist()

            images = []
            print("saving images")
            for idx in tqdm(range(len(self.labels))):
                img_name = os.path.join(f"/home/mil/gupta/ifood18/data/{mode}_set/", self.pic_names[idx])

                image = ndimage.imread(img_name, mode="RGB")
                image = misc.imresize(image, (256, 256), mode='RGB')
                images.append(image)
            self.images = np.array(images, dtype="uint8")
            # save
            np.save(f"/data/ugui0/noguchi/ifood/{mode}_labels.npy", self.labels)
            np.save(f"/data/ugui0/noguchi/ifood/{mode}_label_weight.npy", self.label_weight)
            np.save(f"/data/ugui0/noguchi/ifood/{mode}_pic_names.npy", self.pic_names)
            np.save(f"/data/ugui0/noguchi/ifood/{mode}_images.npy", self.images)
        self.label_weight = self.label_weight.astype("float32")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.flag:
            image = self.augment_images(image)
        correct_label = self.labels[idx]
        # correct_label = correct_label.astype('int')
        if self.transform:
            image = self.transform(image)
        return (image, correct_label, self.label_weight[correct_label])


class H5Dataset(Dataset):
    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        h5_file = h5py.File(file_path)
        self.data = h5_file.get('data')
        self.target = h5_file.get('labels')

    def __getitem__(self, index):
        return (torch.from_numpy(self.data[index, :, :, :]),
                torch.from_numpy(np.array(self.target[index])))

    def __len__(self):
        return self.data.shape[0]


def main():
    global args, best_prec3
    args = parser.parse_args()
    assert args.model in ["resnet152", "se_resnext101_32x4d"]
    if args.tensorboard:
        configure("runs/%s" % (args.name))

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

    transform_test = transforms.Compose([transforms.ToPILImage(),
                                         # transforms.Resize(192,192),
                                         transforms.CenterCrop(224),
                                         # transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor()
                                         # normalize
                                         ])

    assert (args.dataset in ["ifood", "food101N"])

    if args.dataset == "ifood":
        train_data_path = "/home/mil/gupta/ifood18/data/train_set/"
        val_data_path = "/home/mil/gupta/ifood18/data/val_set/"

        train_label = "/home/mil/gupta/ifood18/data/labels/train_info.csv"
        val_label = "/home/mil/gupta/ifood18/data/labels/val_info.csv"
    elif args.dataset == "food101N":
        train_data_path = "/home/mil/gupta/ifood18/data/Food-101N_release/images"
        val_data_path = "/home/mil/gupta/ifood18/data/Food-101N_release/images"
        if not os.path.exists("/home/mil/noguchi/M1/ifood/foodx/data/food101N/train_info.csv"):
            train_data = pd.read_csv("/home/mil/gupta/ifood18/data/Food-101N_release/meta/verified_train.tsv", sep='\t')
            train_data = list(train_data[train_data["verification_label"] == 1]["class_name/key"])
            val_data = pd.read_csv("/home/mil/gupta/ifood18/data/Food-101N_release/meta/verified_val.tsv", sep='\t')
            val_data = list(val_data[val_data["verification_label"] == 1]["class_name/key"])
            train_data += val_data
            class_label = list(set([data.split("/")[0] for data in train_data]))

            train_info = "\n".join([f"{data},{class_label.index(data.split('/')[0])}" for data in train_data])
            with open("/home/mil/noguchi/M1/ifood/foodx/data/food101N/train_info.csv", "w") as f:
                f.write(train_info)

            val_info = "\n".join([f"{data},{class_label.index(data.split('/')[0])}" for data in val_data])
            with open("/home/mil/noguchi/M1/ifood/foodx/data/food101N/val_info.csv", "w") as f:
                f.write(val_info)

        train_label = "/home/mil/noguchi/M1/ifood/foodx/data/food101N/train_info.csv"
        val_label = "/home/mil/noguchi/M1/ifood/foodx/data/food101N/val_info.csv"

    for mode in ["train", "val"]:
        if not os.path.exists(f"/data/ugui0/noguchi/ifood/{mode}_images.npy"):
            csv_file = f"/home/mil/gupta/ifood18/data/labels/{mode}_info.csv"
            data = pd.read_csv(csv_file, header=None, names=["name_of_pic", "noisy_label"])

            labels = data.noisy_label.tolist()
            num_label = np.bincount(labels)
            label_weight = 1 / num_label
            label_weight = label_weight / np.mean(label_weight)
            # read addresses and labels from the 'train' folder
            pic_names = data.name_of_pic.tolist()

            images = []
            print("saving images")
            for idx in tqdm(range(len(labels))):
                img_name = os.path.join(f"/home/mil/gupta/ifood18/data/{mode}_set/", pic_names[idx])

                image = ndimage.imread(img_name, mode="RGB")
                image = misc.imresize(image, (256, 256), mode='RGB')
                images.append(image)
            images = np.array(images, dtype="uint8")
            # save
            np.save(f"/data/ugui0/noguchi/ifood/{mode}_labels.npy", labels)
            np.save(f"/data/ugui0/noguchi/ifood/{mode}_label_weight.npy", label_weight)
            np.save(f"/data/ugui0/noguchi/ifood/{mode}_pic_names.npy", pic_names)
            np.save(f"/data/ugui0/noguchi/ifood/{mode}_images.npy", images)

    if args.debug:
        train_images = np.load(f"/data/ugui0/noguchi/ifood/train_images.npy", mmap_mode='r')
        val_images = np.load(f"/data/ugui0/noguchi/ifood/val_images.npy", mmap_mode='r')
    else:
        t1 = time.time()
        train_images = np.load(f"/data/ugui0/noguchi/ifood/train_images.npy")
        val_images = np.load(f"/data/ugui0/noguchi/ifood/val_images.npy")
        print(f"{time.time() - t1} sec to load images")

    train_dataset = FoodDataset("train", train_images, transform_train, training=True)
    val_dataset = FoodDataset("val", val_images, transform_train)

    # custom_mnist_from_csv = \
    #    CustomDatasetFromCSV('../data/mnist_in_csv.csv',
    #                         28, 28,
    #                         transformations)

    batchsize = 40
    use_BC = args.BC or args.BCp
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchsize * (1 + use_BC),
                                               shuffle=True, num_workers=8)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batchsize, num_workers=8)

    # 	train_labels = pd.read_csv('./data/labels/train_info.csv')







    # 	train_loader = torch.utils.data.DataLoader(
    #         datasets.__dict__[args.dataset.upper()](train_data_path, train=True, download=True,
    #                          transform=transform_train),
    #         batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = torch.utils.data.DataLoader(
    #         datasets.__dict__[args.dataset.upper()](val_data_path, train=False, transform=transform_test),
    #         batch_size=args.batch_size, shuffle=True, **kwargs)






    # create model
    # model = WideResNet(args.layers, 211, args.widen_factor, dropRate=args.droprate)

    # model = pretrainedmodels.__dict__["resnet152"](num_classes=1000, pretrained='imagenet')se_resnext101_32x4d
    model_one = pretrainedmodels.__dict__[args.model](num_classes=1000, pretrained='imagenet')
    model_one = finetune.FineTuneModel(model_one)
    model_one.train_params()
    model_two = pretrainedmodels.__dict__[args.model](num_classes=1000, pretrained='imagenet')
    model_two = finetune.FineTuneModel(model_two)
    model_two.train_params()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model_one.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    model_one = torch.nn.DataParallel(model_one).cuda()
    model_two = torch.nn.DataParallel(model_one).cuda()
    # model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec3 = checkpoint['best_prec3']
            model_one.load_state_dict(checkpoint['state_dict_one'])
            model_two.load_state_dict(checkpoint['state_dict_two'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.KLDivLoss(size_average=False).cuda()
    optimizer = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model_one.parameters())) +
                                list(filter(lambda p: p.requires_grad, model_two.parameters())),
                                args.lr,
                                momentum=args.momentum,  # nesterov=args.nesterov,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #                              args.lr,
    #                              weight_decay=args.weight_decay)
    prec3 = 0
    for epoch in range(args.start_epoch, args.epochs):
        # if epoch == 0:
        #     # update all parameters from epoch1
        #     model.train_params()
        #     optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
        #                                 momentum=args.momentum,  # nesterov=args.nesterov,
        #                                 weight_decay=args.weight_decay)

        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        train(train_loader, model_one, model_two, criterion, optimizer, epoch)

        if epoch % args.validate_freq == 0:
            # evaluate on validation set
            prec3 = validate(val_loader, model_one, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec3 > best_prec3
        best_prec3 = max(prec3, best_prec3)
        save_checkpoint({
            'epoch': epoch + 1,
            'model_one': model_one,
            'model_two': model_two,
            'state_dict_one': model_one.state_dict(),
            'state_dict_two': model_two.state_dict(),
            'best_prec3': best_prec3,
        }, is_best)

    print('Best accuracy: ', best_prec3)


def train(train_loader, model_one, model_two, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""

    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    batch_time = AverageMeter()
    losses = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model_one.train()
    model_two.train()

    end = time.time()
    for i, (inp, target, class_weight) in enumerate(train_loader):
        target = target.cuda(async=True)
        inp = inp.cuda()
        class_weight = class_weight.cuda()
        if args.BC:
            # mix
            target = target.type(torch.LongTensor)
            batchsize = inp.shape[0] // 2
            rand = torch.rand(batchsize).cuda()
            inp = inp[:batchsize] * rand.view(-1, 1, 1, 1) + inp[-batchsize:] * (1 - rand.view(-1, 1, 1, 1))
            eye = torch.eye(211).cuda()
            target = eye[target[:batchsize]] * rand.view(-1, 1) + eye[target[-batchsize:]] * (1 - rand.view(-1, 1))
        elif args.BCp:
            # mix
            target = target.type(torch.LongTensor)
            batchsize = inp.shape[0] // 2
            rand = torch.rand(batchsize).cuda()

            mean = torch.mean(inp.view(inp.shape[0], -1), dim=1)
            sigma = torch.std(inp.view(inp.shape[0], -1), dim=1)
            p = 1 / (1 + sigma[:batchsize] / sigma[-batchsize:] * (1 - rand) / rand)
            inp = inp - mean.view(-1, 1, 1, 1)
            inp = inp[:batchsize] * p.view(-1, 1, 1, 1) + inp[-batchsize:] * (1 - p.view(-1, 1, 1, 1))
            inp = inp / torch.sqrt(p ** 2 + (1 - p) ** 2).view(-1, 1, 1, 1)
            eye = torch.eye(211).cuda()
            target = eye[target[:batchsize]] * rand.view(-1, 1) + eye[target[-batchsize:]] * (1 - rand.view(-1, 1))

        else:
            batchsize = inp.shape[0]
            target = target.type(torch.LongTensor)
            eye = torch.eye(211)
            target = eye[target]

        input_var = torch.autograd.Variable(inp)
        target_var = torch.autograd.Variable(target)

        # compute output
        output1 = model_one(input_var)
        output2 = model_one(input_var)

        if args.class_weight:
            loss = (target_var * (torch.log(target_var + 1e-10) - F.log_softmax(output, 1))).sum(1) * \
                   class_weight[:batchsize] * class_weight[-batchsize:]
            loss = loss.mean()

        else:
            loss = criterion(F.log_softmax(output1, 1), target_var) / batchsize
            loss += criterion(F.log_softmax(output2, 1), target_var) / batchsize

        diff = torch.mean(kl_loss(F.softmax(output1), F.softmax(output2)) +
                          kl_loss(F.softmax(output2), F.softmax(output1)))
        loss += diff
        prec3 = accuracy(output1.data, target,
                         topk=(1, 3))  ## res is of shape [2,B] representing top 1 and top k accuracy respectively
        losses.update(loss.data[0], inp.size(0))
        top3.update(prec3, inp.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top3=top3))


def kl_loss(source, target):
    batchsize = source.shape[0]
    loss = (target * (torch.log(target + 1e-10) - F.log_softmax(source, 1))).sum() / batchsize
    return loss


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (input, target, class_weight) in enumerate(val_loader):
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
        class_weight = class_weight.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        with torch.no_grad():
            output = model(input_var)

        if args.class_weight:
            loss = (target_var * (torch.log(target_var + 1e-10) - F.log_softmax(output, 1))).sum(1) * \
                   class_weight[:batchsize] * class_weight[-batchsize:]
            loss = loss.mean()
        else:
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
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top3.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = args.lr * (
        (0.2 ** int(epoch >= 100)) * (0.2 ** int(epoch >= 120)) * (0.2 ** int(epoch >= 160)) * (
            0.2 ** int(epoch >= 200)))
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
    # torch.multiprocessing.set_start_method("spawn", force=True)
    main()
