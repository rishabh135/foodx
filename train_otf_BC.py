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
parser.add_argument('--name', default='WideResNet-ifood-28-4-otf-BC-aug', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')

parser.add_argument('--validate_freq', '-valf', default=2, type=int,
                    help='validate frequency (default: 100)')

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


# print(type(tmp))
# grid = seq.draw_grid(image, cols=8, rows=8)
# misc.imsave("examples_grid.jpg", grid)
# io.imsave("examples_image.jpg", tmp)



# return tmp




class FoodDataset(Dataset):
    """Food dataset."""

    def __init__(self, root_dir, csv_file, transform, training=False):
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
        image = misc.imresize(image, (192, 192), mode='RGB')
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
            transforms.RandomCrop(128),
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
                                         transforms.CenterCrop(128),
                                         # transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor()
                                         # normalize
                                         ])

    kwargs = {'num_workers': 4, 'pin_memory': True}
    # assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')

    train_data_path = "/home/mil/gupta/ifood18/data/train_set/"
    val_data_path = "/home/mil/gupta/ifood18/data/val_set/"

    train_label = "/home/mil/gupta/ifood18/data/labels/train_info.csv"
    val_label = "/home/mil/gupta/ifood18/data/labels/val_info.csv"

    # transformations = transforms.Compose([transforms.ToTensor()])


    # train_h5 = "/home/mil/gupta/ifood18/data/h5data/train_data.h5py"
    train_h5 = "/home/mil/gupta/ifood18/data/h5data/train_data_iteration_1.h5py"
    val_h5 = "/home/mil/gupta/ifood18/data/h5data/val_data_iteration_1.h5py"

    # 	train_dataset =  H5Dataset(train_h5)
    # 	val_dataset =  H5Dataset(val_h5)

    train_dataset = FoodDataset(train_data_path, train_label, transform_train, training=True)
    val_dataset = FoodDataset(val_data_path, val_label, transform_test)

    # custom_mnist_from_csv = \
    #    CustomDatasetFromCSV('../data/mnist_in_csv.csv',
    #                         28, 28,
    #                         transformations)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=16)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=128, num_workers=16)

    # 	train_labels = pd.read_csv('./data/labels/train_info.csv')







    # 	train_loader = torch.utils.data.DataLoader(
    #         datasets.__dict__[args.dataset.upper()](train_data_path, train=True, download=True,
    #                          transform=transform_train),
    #         batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = torch.utils.data.DataLoader(
    #         datasets.__dict__[args.dataset.upper()](val_data_path, train=False, transform=transform_test),
    #         batch_size=args.batch_size, shuffle=True, **kwargs)






    # create model
    model = WideResNet(args.layers, 211, args.widen_factor, dropRate=args.droprate)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec3 = checkpoint['best_prec3']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.KLDivLoss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum, nesterov=args.nesterov,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        if epoch % args.validate_freq == 0:
            # evaluate on validation set
            prec3 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec3 > best_prec3
        best_prec3 = max(prec3, best_prec3)
        save_checkpoint({
            'epoch': epoch + 1,
            "model": model,
            'state_dict': model.state_dict(),
            'best_prec3': best_prec3,
        }, is_best)

    print('Best accuracy: ', best_prec3)


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""

    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    batch_time = AverageMeter()
    losses = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inp, target) in enumerate(train_loader):
        # mix
        target = target.type(torch.LongTensor)
        batchsize = inp.shape[0] // 2
        rand = torch.rand(batchsize)
        inp = inp[:batchsize] * rand.view(-1, 1, 1, 1) + inp[-batchsize:] * (1 - rand.view(-1, 1, 1, 1))
        eye = torch.eye(211)
        target = eye[target[:batchsize]] * rand.view(-1, 1) + eye[target[-batchsize:]] * (1 - rand.view(-1, 1))

        target = target.cuda(async=True)
        inp = inp.cuda()

        input_var = torch.autograd.Variable(inp)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(F.log_softmax(output, 1), target_var)

        prec3 = accuracy(output.data, target,
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
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        # mix
        target = target.type(torch.LongTensor)
        eye = torch.eye(211)
        target = eye[target]

        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(F.log_softmax(output, 1), target_var)

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
    lr = args.lr * ((0.2 ** int(epoch >= 120)) * (0.2 ** int(epoch >= 240)) * (0.2 ** int(epoch >= 320)))
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
