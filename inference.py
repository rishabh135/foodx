
from __future__ import print_function, division
import os

import matplotlib
matplotlib.use('Agg')
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import pandas as pd
from skimage import io, transform
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from tqdm import tqdm
import cv2,sys
from PIL import Image
		
from scipy import ndimage, misc
from skimage import data, transform


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
import numpy as np
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
#import torch
#import torch.multiprocessing
#torch.multiprocessing.set_start_method('spawn')

import glob

import datetime



import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch ifood18 inferencing')


parser.add_argument('--save-file', default='/home/mil/gupta/ifood18/runs/', type=str,
					help='location of saved pt model file to load weights from')


parser.add_argument('--test-data', default='/home/mil/gupta/ifood18/data/test_set/', type=str,
					help='location of images to run inference on')

parser.add_argument('--output-text', default='/home/mil/gupta/ifood18/data/result_test_on_noguchi.txt', type=str,
					help='location of images to run inference on')



#### Only change this to the folder of your choice
parser.add_argument('--name-save-file', default='WideResNet-ifood-28-4-otf-adam-with-augmentation/', type=str,
					help='name of run')


# /home/mil/noguchi/M1/ifood/foodx/runs/ResNet152-ifood-28-4-otf-BC-aug_2/model_best.pth.tar

parser.add_argument('--layers', default=28, type=int,
					help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=4, type=int,
					help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.5, type=float,
					help='dropout probability (default: 0.0)')




parser.add_argument('--batch-size', default=128, type=int,
					help='batch size for inferencing (default: 128)')

parser.add_argument('--class-size', default=211, type=int,
					help='class size depending upon dataset  (default: 211 for ifood dataset)')


parser.add_argument('--resize-size', default=256, type=int,
					help='resize image size depending upon dataset  (default: 256 for ifood dataset)')

parser.add_argument('--crop-size', default=224, type=int,
					help='size of image to be input for the model  (default: 224 for resnet 152 model)')




args = parser.parse_args()

## load
#model = Model() # the model should be defined with the same code you used to create the trained model


args.save_file = "/home/mil/noguchi/M1/ifood/foodx/runs/"

args.name_save_file = "ResNet152-ifood-28-4-otf-BC-aug_2/"







class FoodDatasetTest(Dataset):
	"""Food dataset."""
	def __init__(self, root_dir, csv_file, transform=None , training = False):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		# self.labels = pd.read_csv(csv_file)
		
		data = pd.read_csv(csv_file, header=None , names= ["name_of_pic"])
		#print(selfdata.head())
		# read addresses and labels from the 'train' folder
		self.pic_names = data.name_of_pic.tolist()
		# print("Length of val data is : ",len(val_data))
		
		self.root_dir = root_dir
		self.flag = training
		self.transform = transform

	def __len__(self):
		return len(self.pic_names)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir,self.pic_names[idx])
		name = self.pic_names[idx]

		image = ndimage.imread(img_name, mode="RGB")
		image = misc.imresize(image, (args.resize_size,args.resize_size),mode='RGB')
		#image = transform.resize(image , (128,128))
		# 
		#image = resizeimage.resize_cover(Image.open(img_name), [128, 128])
		
		if(self.flag == True):
			image = augment_images(image)
		

		#correct_label = self.labels[idx]
		#correct_label = correct_label.astype('int')
		if self.transform:
			image = self.transform(image)
		
		
		#plt.imsave("examples_image_after_trabsform.jpg", image.numpy().transpose(1,2,0))
		#print(image.size)
		#sys.exit()
		return (image , name)



	
	
transform_test = transforms.Compose([
		
		transforms.ToPILImage(),
		#transforms.Resize(192,192),
		# transforms.CenterCrop(128),
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor()
		#normalize
		])
	
	




model_path = os.path.join(args.save_file, args.name_save_file)
print("loading model from ", model_path)




# create model important it has same architecture as the original model
#model = WideResNet(args.layers, args.class_size, args.widen_factor, dropRate=args.droprate)

# get the number of model parameters
# for training on multiple GPUs.
# Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
best_model = torch.load( model_path + "model_best.pth.tar")








model = best_model['model']
#model = torch.nn.DataParallel(model).cuda()
#model.load_state_dict(best_model['state_dict'])



## most recent model
#checkpoint = torch.load( model_path + "checkpoint.pth.tar")


#import itertools
# a = [["a","b"], ["c"]]




print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

now = datetime.datetime.now()



def center_crop(x, center_crop_size, **kwargs):
	#print(x.shape)
	centerw, centerh = x.shape[2]//2, x.shape[3]//2
	halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
	return x[:, :, centerw-halfw:centerw+halfw,centerh-halfh:centerh+halfh]



def imsave(file_name, img):
    """
    save a torch tensor as an image
    :param file_name: 'image/folder/image_name'
    :param img: 3*h*w torch tensor
    :return: nothing
    """
    assert(type(img) == torch.FloatTensor,
           'img must be a torch.FloatTensor')
    ndim = len(img.size())
    assert(ndim == 2 or ndim == 3,
           'img must be a 2 or 3 dimensional tensor')
    img = img.numpy()
    if ndim == 3:
        plt.imsave(file_name, np.transpose(img, (1, 2, 0)))
    else:
        plt.imsave(file_name, img, cmap='gray') 



# def predict_10_crop(img, ix, top_n=3, plot=True, preprocess=False, debug=False):
# 	flipped_X = np.fliplr(img)
# 	crop_size = 128
# 	crops = [
# 		img[:, :crop_size,:crop_size], # Upper Left
# 		img[:,:crop_size, img.shape[2]-crop_size:], # Upper Right
# 		img[:,img.shape[1]-crop_size:, :crop_size], # Lower Left
# 		img[:,img.shape[1]-crop_size:, img.shape[2]-crop_size:], # Lower Right
# 		center_crop(img, (crop_size, crop_size)),
		
# 		#flipped_X[:299,:299, :],
# 		#flipped_X[:299, flipped_X.shape[1]-299:, :],
# 		#flipped_X[flipped_X.shape[0]-299:, :299, :],
# 		#flipped_X[flipped_X.shape[0]-299:, flipped_X.shape[1]-299:, :],
# 		#center_crop(flipped_X, (299, 299))
# 	]
# 	#if preprocess:
# 	#	crops = [preprocess_input(x.astype('float32')) for x in crops]

# #	if plot:
# # 		fig, ax = plt.subplots(2, 5, figsize=(10, 4))
# # 		ax[0][0].imshow(crops[0].transpose(1,2,0))
# # 		ax[0][1].imshow(crops[1])
# # 		ax[0][2].imshow(crops[2])
# # 		ax[0][3].imshow(crops[3])
# # 		ax[0][4].imshow(crops[4])
# # 		ax[1][0].imshow(crops[0])
# # 		ax[1][1].imshow(crops[1])
# # 		ax[1][2].imshow(crops[2])
# # 		ax[1][3].imshow(crops[3])
# # 		ax[1][4].imshow(crops[4])
# # 		fig.savefig(args.save_file + "test_crops.jpg")
# 	#for temp in crops:
# 	#	print(temp.shape)
	
# 	crop = torch.cat(crops, dim=0)
	
# 	print(crop.shape)
# 	output = model(crop) 
# 	prob , y_pred = output.data.topk(3, 1, True, True) 
# 	#y_pred = model(crops)
# 	preds = np.argmax(y_pred, axis=1)
# 	top_n_preds= np.argpartition(y_pred, -top_n)[:,-top_n:]
# 	if debug:
# 		print('Top-1 Predicted:', preds)
# 		print('Top-3 Predicted:', top_n_preds)
# 		#print('True Label:', y_test[ix])
# 	return preds, top_n_preds






def batch_prediction(image_batch, model):
	preds_10_crop = {}
	##print(image_batch.shape)
	#itt = input()
	
	
	
	crop_size = args.crop_size
	
	crops = [
		image_batch[:, :, :crop_size,:crop_size], # Upper Left
		image_batch[:, :, :crop_size, image_batch.shape[3]-crop_size:], # Upper Right
		image_batch[:, :, image_batch.shape[2]-crop_size:, :crop_size], # Lower Left
		image_batch[:, :, image_batch.shape[2]-crop_size:, image_batch.shape[3]-crop_size:], # Lower Right
		center_crop(image_batch, (crop_size, crop_size)),
		
		#flipped_X[:299,:299, :],
		#flipped_X[:299, flipped_X.shape[1]-299:, :],
		#flipped_X[flipped_X.shape[0]-299:, :299, :],
		#flipped_X[flipped_X.shape[0]-299:, flipped_X.shape[1]-299:, :],
		#center_crop(flipped_X, (299, 299))
	]
	
	
	print("shape of crop for first image ",crops[0].shape)
	
	prob_classes = []
	y_pred_classes = []
	
	final_prob = torch.zeros([args.batch_size, args.class_size]).cuda()
	for iy in range(len(crops)):
		with torch.no_grad():
			output = model(crops[iy])
		
		prob , y_pred = output.data.topk(3, 1, True, True) 
		output_controlled = torch.zeros(output.data.shape).cuda()
		output_controlled[np.arange(args.batch_size*3)// 3, y_pred.view(-1)] = prob.view(-1)
		
		
		#itt = input()
		prob_classes.append(prob)
		y_pred_classes.append(y_pred)
		
		final_prob = final_prob + output_controlled
	
	
	prob , y_pred = final_prob.data.topk(3, 1, True, True) 
# 	print(prob)
# 	print(y_pred)
# 	print(y_pred.shape)
	
# 	print(prob_classes[0].shape)
# 	itt = input()
	
# 	# tmp = (itertools.chain.from_iterable(prob_classes))
# 	# print (tmp.shape)
	
# 	combining_prob = torch.cat(prob_classes,dim=1)
# 	combining_classes = torch.cat(y_pred_classes,dim=1)
	# combining_prob = list(zip(f) for f in prob_classes)
	# combining_ind = list(zip(f) for f in y_pred_classes)
	#print(combining_prob.shape)
	
	
	# print("important : ", combining_prob[0][0].shape)
	# print(combining_prob[0])
	
# 	preds = np.argmax(y_pred, axis=1)
	
	#top_n_preds= np.argpartition(combining_classes, -top_n)[:,-top_n:]
	#print(top_n_preds)
# 	if debug:
# 		print('Top-1 Predicted:', preds)
# 		print('Top-3 Predicted:', top_n_preds)
# 		#print('True Label:', y_test[ix])
	
	
	
# 	for ix in range(len(image_batch)):
# 		if ix % args.batch_size-1 == 0:
# 			print("Completed batch ",ix)
# 		preds_10_crop[ix] = predict_10_crop(image_batch[ix], ix)
		

# 	preds_uniq = {k: np.unique(v[0]) for k, v in preds_10_crop.items()}
# 	preds_hist = np.array([len(x) for x in preds_uniq.values()])

# 	plt.hist(preds_hist, bins=11)
# 	plt.title('Number of unique predictions per image')

	
	
	
	return y_pred
	
	
	
	
	
	
	
		
def test(test_loader, model):
	"""Perform test on the test set"""

	
	global args
	# switch to evaluate mode
	model.eval()

	end = time.time()
	
	with open(args.output_text, "w") as file:
		
		file.write("id,predicted\n")

		for i, (inp,name) in tqdm(enumerate(test_loader)):

			#target = target.type(torch.LongTensor).cuda(async=True)
			#inp = inp.cuda()
			input_var = torch.autograd.Variable(inp, volatile=True)
			#target_var = torch.autograd.Variable(target, volatile=True)

			pred = batch_prediction(input_var , model)
			
			#for range 
			#preds, top_n_preds 
			
			# compute output
			#output = model(input_var)
			#_, pred = output.data.topk(3, 1, True, True)
			

			#test_000000.jpg,127 121 99
			#loss = criterion(output, target_var)
			for row in range(len(pred)):
				
				file.write("%s,%d %d %d\n"%(name[row],pred[row][0],pred[row][1],pred[row][2]))
				# measure accuracy and record loss
				#prec3 = accuracy(output.data, target, topk=(1,3))
				#losses.update(loss.data[0], inp.size(0))
				#top3.update(prec3, inp.size(0))

				# measure elapsed time
				#batch_time.update(time.time() - end)



	print("Completed testing and saved the corresponsding file")
	print("Took time : ", time.time()-end)
	return 
		

	
	
	
	
	
	
	
	
	
	
test_data_path ="/home/mil/gupta/ifood18/data/test_set/"
	
test_csv ="/home/mil/gupta/ifood18/data/labels/test_info.csv"

test_dataset = FoodDatasetTest(test_data_path, test_csv, transform = transform_test )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size , shuffle = False, num_workers=4)


# train for one epoch
test(test_loader, model)
