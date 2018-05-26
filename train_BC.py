
from __future__ import print_function, division
import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from tqdm import tqdm

from PIL import Image
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
# import torch.utils.data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from torch.autograd import Variable

from wideresnet import WideResNet

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value
#import torch
#import torch.multiprocessing
#torch.multiprocessing.set_start_method('spawn')

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='ci', type=str,
					help='dataset (ifood18[default] or cifar10 or cifar100)')
parser.add_argument('--epochs', default=200, type=int,
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
					help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
					help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
					help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
					help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
					help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=4, type=int,
					help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.3, type=float,
					help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
					help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
					help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-ifood-28-10', type=str,
					help='name of experiment')
parser.add_argument('--tensorboard',
					help='Log progress to TensorBoard', action='store_true')

parser.add_argument('--validate_freq', '-valf', default=200, type=int,
					help='validate frequency (default: 100)')


parser.set_defaults(augment=True)

best_prec3 = 0






class FoodDataset(Dataset):
	"""Food dataset."""
	def __init__(self, root_dir, csv_file, transform):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.labels = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir,self.labels.iloc[idx, 0])
		image = Image.open(img_name)
		correct_label = self.labels.iloc[idx, 1]
		#correct_label = correct_label.astype('int')
		if self.transform:
			image = self.transform(image)

		sample = (image, correct_label)

		return sample




	
	
class H5Dataset(Dataset):

	def __init__(self, file_path):
		super(H5Dataset, self).__init__()
		h5_file = h5py.File(file_path)
		self.data = h5_file.get('data')
		self.target = h5_file.get('labels')
		#f = h5py.File("foo.hdf5", "r")
		#set1 = np.asarray(h5_file["foo/bar"])
		#self.data = h5_file["data"]
		#self.labels = h5_file["labels"]
		#print(self.data.shape)
		#print(type(self.data))
		#var3 = set1["var3"]
		
		
	def __getitem__(self, index): 
        batchsize = len(index) / 2
        r = np.random.rand(batchsize).astype("float32")
		return (torch.from_numpy(self.data[index[:batchsize],:,:,:] * r + self.data[index[-batchsize:],:,:,:] * (1-r)),
				torch.from_numpy((np.eye(211)[np.array(self.target[index[:batchsize]])] * r + np.eye(211)[np.array(self.target[index[:batchsize]])] * (1-r)).astype("int32")))

	def __len__(self):
		return self.data.shape[0]










def main():
	global args, best_prec1
	args = parser.parse_args()
	if args.tensorboard: 
		configure("runs/%s"%(args.name))

		
		
		
		
	# Data loading code
	# normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
	#								 std=[x/255.0 for x in [63.0, 62.1, 66.7]])

# 	if args.augment:
# 		transform_train = transforms.Compose([
# 			transforms.Resize(256,256),
# 			transforms.ToTensor(),
# 			transforms.Lambda(lambda x: F.pad(
# 								Variable(x.unsqueeze(0), requires_grad=False, volatile=True),
# 								(4,4,4,4),mode='reflect').data.squeeze()),
# 			transforms.ToPILImage(),
# 			transforms.RandomCrop(32),
# 			transforms.RandomHorizontalFlip(),
# 			transforms.ToTensor(),
# 			normalize,
# 			])
# 	else:
# 		transform_train = transforms.Compose([
# 			transforms.ToTensor(),
# 			normalize,
# 			])
	
# 	transform_test = transforms.Compose([
# 		transforms.ToTensor(),
# 		normalize
# 		])

	
	
	
	#kwargs = {'num_workers': 1, 'pin_memory': True}
	#assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
	
	train_data_path ="/home/mil/gupta/ifood18/data/training_set/train_set/"
	val_data_path ="/home/mil/gupta/ifood18/data/val_set/"
	
	train_label ="/home/mil/gupta/ifood18/data/labels/train_info.csv"
	val_label ="/home/mil/gupta/ifood18/data/labels/val_info.csv"
	
	
	#transformations = transforms.Compose([transforms.ToTensor()])

	
	# train_h5 = "/home/mil/gupta/ifood18/data/h5data/train_data.h5py"
	train_h5 =  "/home/mil/gupta/ifood18/data/h5data/train_data_partial.h5py"
	train_dataset =  H5Dataset(train_h5)
	#val_dataset =  H5Dataset(val_data_path, val_label, transform= None)
	

	

	#custom_mnist_from_csv = \
	#    CustomDatasetFromCSV('../data/mnist_in_csv.csv',
	#                         28, 28,
	#                         transformations)

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle= True, num_workers=1)

	#val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=256,num_workers=1,shuffle=True)
	
	
	
	
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
	print('Number of model parameters: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))

	# for training on multiple GPUs.
	# Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
	# model = torch.nn.DataParallel(model).cuda()
	model = model.cuda()

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	# define loss function (criterion) and optimizer
	criterion = nn.KLDivLoss().cuda()
	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum, nesterov = args.nesterov,
								weight_decay=args.weight_decay)

	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch+1)

		# train for one epoch
		train(train_loader, model, criterion, optimizer, epoch)

		
		#if epoch % args.validate_freq == 0:
			# evaluate on validation set
		prec3 = validate(val_loader, model, criterion, epoch)

		# remember best prec@1 and save checkpoint
		is_best = prec3 > best_prec3
		best_prec1 = max(prec3, best_prec3)
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_prec1': best_prec3,
		}, is_best)
	
	print ('Best accuracy: ', best_prec1)

def train(train_loader, model, criterion, optimizer, epoch):
	"""Train for one epoch on the training set"""   
	
	
# 	print(torch.cuda.current_device())
# 	print(torch.cuda.device_count())
# 	print(torch.cuda.get_device_name(0))
	
	
	
	
	
	batch_time = AverageMeter()
	losses = AverageMeter()
	top3 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	#print(train_loader)
	#print(size(train_loader))
	for i, (inp, target) in enumerate(train_loader):
		
		
# 		print("target is : ",target)
# 		print(target.shape)
		
# 		#print("\n\n\n input is : ",inp)
# 		print(inp.shape)
		
		
		
# 		#t =  input()
		
		print(inp.shape)
		tttt =input()
		target = target.type(torch.LongTensor).cuda(async=True)
		inp = inp.cuda()
		
		input_var = torch.autograd.Variable(inp)
		target_var = torch.autograd.Variable(target)
		
		# compute output
		output = model(input_var)
		print("output shape: ",output.shape)
		loss = criterion(output, target_var)
		# print(loss)
		# print(loss.shape)
		# print(loss.data)
		# print("loss is fine")
		#idsds =input()
		# measure accuracy and record loss
		prec3 = accuracy(output.data, target, topk=(1,3))[-1] ## res is of shape [2,B] representing top 1 and top k accuracy respectively
		losses.update(loss.data[0], inp.size(0))
		top3.update(prec3[0], inp.size(0))
		
		# print(prec3[0])
		# print(top3.debug())
		# print("**"*5)
		# intttt = input()
		# compute gradient and do SGD step
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
				  'Prec@1 {top3.val:.3f} ({top3.avg:.3f})'.format(
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
		target = target.cuda(async=True)
		input = input.cuda()
		input_var = torch.autograd.Variable(input, volatile=True)
		target_var = torch.autograd.Variable(target, volatile=True)

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		prec3 = accuracy(output.data, target, topk=(1,3))[-1]
		
		
		losses.update(loss.data[0], input.size(0))
		top3.update(prec3[0], input.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Test: [{0}/{1}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top3.val:.3f} ({top3.avg:.3f})'.format(
					  i, len(val_loader), batch_time=batch_time, loss=losses,
					  top3=top3))

	print(' * Prec@1 {top3.avg:.3f}'.format(top3=top3))
	# log to TensorBoard
	if args.tensorboard:
		log_value('val_loss', losses.avg, epoch)
		log_value('val_acc', top1.avg, epoch)
	return top3.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	"""Saves checkpoint to disk"""
	directory = "runs/%s/"%(args.name)
	if not os.path.exists(directory):
		os.makedirs(directory)
	filename = directory + filename
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

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
		
	def debug(self):
		print("val = ",self.val)
		print("sum = ",self.sum)
		print("count = ",self.count)
		print("avg = ",self.avg)
		
		
		


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
	lr = args.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120))* (0.2 ** int(epoch >= 160)))
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

	print("Output is : ", output)
	print(output.shape)
	_, pred = output.topk(maxk, 1, True, True)
	
	print(pred)
	print(pred.shape)
	i =input()
	
	pred = pred.t()
	print("target" , target.view(1, -1).expand_as(pred))
	print(target.shape)
	print("\n\n\n\n")
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	print(correct)
	print(correct.shape)
	inp3 =input() 
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		# print("inside for loop : ",correct[:k].view(-1).shape)
		#inp4 = input()
		res.append(correct_k.mul_(100.0 / batch_size))
		
		
	print(res)
	print("\n\n")
	print(res[0])
	print("all done")
	ii =input()
	return res

if __name__ == '__main__':
	
	# torch.multiprocessing.set_start_method("spawn", force=True)
	main()
