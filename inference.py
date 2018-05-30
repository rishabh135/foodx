
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
import cv2,sys
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
#import torch
#import torch.multiprocessing
#torch.multiprocessing.set_start_method('spawn')

import glob

import datetime



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
#model = Model() # the model should be defined with the same code you used to create the trained model








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
		image = misc.imresize(image, (192,192),mode='RGB')
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



	
	
transform_test = transforms.Compose([transforms.ToPILImage(),
		#transforms.Resize(192,192),
		transforms.CenterCrop(128),
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor()
		#normalize
		])
	
	

# create model
model = WideResNet(args.layers, 211, args.widen_factor, dropRate=args.droprate)

# get the number of model parameters
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# for training on multiple GPUs.
# Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
model = torch.nn.DataParallel(model).cuda()



model_path = os.path.join(args.save_file, args.name_save_file)
print("loading model from ", model_path)
state_dict = torch.load( model_path + "model_best.pth.tar")
checkpoint = torch.load( model_path + "checkpoint.pth.tar")

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

		for i, (inp,name) in tqdm(enumerate(test_loader)):

			#target = target.type(torch.LongTensor).cuda(async=True)
			#inp = inp.cuda()
			input_var = torch.autograd.Variable(inp, volatile=True)
			#target_var = torch.autograd.Variable(target, volatile=True)

			# compute output
			output = model(input_var)
			_, pred = output.data.topk(3, 1, True, True)
			

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
	return 0
		
test_data_path ="/home/mil/gupta/ifood18/data/test_set/"
	
test_csv ="/home/mil/gupta/ifood18/data/labels/test_info.csv"

		
		
		
		
test_dataset = FoodDatasetTest(test_data_path, test_csv, transform = transform_test )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128 , shuffle = False, num_workers=4)


# train for one epoch
temp = test(test_loader, model)
