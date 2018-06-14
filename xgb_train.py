from __future__ import print_function, division
from __future__ import unicode_literals
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


import pickle
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor





parser = argparse.ArgumentParser(description='PyTorch ifood18 inferencing')


# parser.add_argument('--save-file', default='/home/mil/gupta/ifood18/runs/', type=str,
# 					help='location of saved pt model file to load weights from')


# parser.add_argument('--test-data', default='/home/mil/gupta/ifood18/data/test_set/', type=str,
# 					help='location of images to run inference on')

# parser.add_argument('--output-text', default='/home/mil/gupta/ifood18/data/result_test_on_noguchi.txt', type=str,
# 					help='location of images to run inference on')



# #### Only change this to the folder of your choice
# parser.add_argument('--name-save-file', default='WideResNet-ifood-28-4-otf-adam-with-augmentation/', type=str,
# 					help='name of run')


# # /home/mil/noguchi/M1/ifood/foodx/runs/ResNet152-ifood-28-4-otf-BC-aug_2/model_best.pth.tar

# parser.add_argument('--layers', default=28, type=int,
# 					help='total number of layers (default: 28)')
# parser.add_argument('--widen-factor', default=4, type=int,
# 					help='widen factor (default: 10)')
# parser.add_argument('--droprate', default=0.5, type=float,
# 					help='dropout probability (default: 0.0)')




parser.add_argument('--batch-size', default=128, type=int,
					help='batch size for inferencing (default: 128)')

parser.add_argument('--class-size', default=211, type=int,
					help='class size depending upon dataset  (default: 211 for ifood dataset)')


parser.add_argument('--resize-size', default=256, type=int,
					help='resize image size depending upon dataset  (default: 256 for ifood dataset)')

parser.add_argument('--crop-size', default=224, type=int,
					help='size of image to be input for the model  (default: 224 for resnet 152 model)')




args = parser.parse_args()



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
			iaa.Fliplr(0.8), # horizontally flip 50% of all images
			iaa.Flipud(0.5), # vertically flip 20% of all images
			# crop images by -5% to 10% of their height/width
			#sometimes(iaa.CropAndPad(
			#	percent=(-0.05, 0.1),
			#	pad_mode=ia.ALL,
			#	pad_cval=(0, 255)
			#)),
			sometimes(iaa.Affine(
				scale={"x": (0.6, 1.4), "y": (0.6, 1.4)}, # scale images to 80-120% of their size, individually per axis
				translate_percent={"x": (-0.4, 0.4), "y": (-0.4, 0.4)}, # translate by -20 to +20 percent (per axis)
				rotate=(-90, 90), # rotate by -45 to +45 degrees
				shear=(-20, 20), # shear by -16 to +16 degrees
				order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
				cval=(0, 255), # if mode is constant, use a cval between 0 and 255
				mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
			)),
			# execute 0 to 5 of the following (less important) augmenters per image
			# don't execute all of them, as that would often be way too strong
			iaa.SomeOf((0, 5),
				[
					sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
					iaa.OneOf([
						iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
						iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
						iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
					]),
					iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
					iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
					# search either for all edges or for directed edges,
					# blend the result with the original image using a blobby mask
					#iaa.SimplexNoiseAlpha(iaa.OneOf([
					#	iaa.EdgeDetect(alpha=(0.5, 1.0)),
					#	iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
					# ])),
					iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
					#iaa.OneOf([
					#	iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
					#	iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
					#]),
					iaa.Invert(0.05, per_channel=True), # invert color channels
					iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
					iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
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
					iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
					iaa.Grayscale(alpha=(0.0, 1.0)),
					sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
					sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
					sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
				],
				random_order=True
			)
		],
		random_order=True
	)
	
	
	#print("type ", type(image))
	
	return seq.augment_image(image)
	
	#print(type(tmp))
	#grid = seq.draw_grid(image, cols=8, rows=8)
	#misc.imsave("examples_grid.jpg", grid)
	#io.imsave("examples_image.jpg", tmp)
	
	

	#return tmp



	
	
	
	
	

# if args.augment:
transform_train = transforms.Compose([
	#transforms.ToTensor(),
	#transforms.Lambda(lambda x: F.pad(
	#					Variable(x.unsqueeze(0), requires_grad=False, volatile=True),
	#					(4,4,4,4),mode='reflect').data.squeeze()),
	transforms.ToPILImage(),
	# transforms.Resize(192,192),
	# transforms.RandomCrop(128),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor()
	#normalize,
	])
# else:
# transform_train = transforms.Compose([
# 	#transforms.Resize(256,256),
# 	#transforms.ToTensor()
# 	#normalize,
# 	])

transform_test = transforms.Compose([transforms.ToPILImage(),
#transforms.Resize(192,192),
#transforms.CenterCrop(128),
#transforms.RandomHorizontalFlip(),
transforms.ToTensor()
#normalize
])



class FoodDataset(Dataset):
	"""Food dataset."""
	def __init__(self, root_dir, csv_file, transform , training = False):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		# self.labels = pd.read_csv(csv_file)
		
		data = pd.read_csv(csv_file, header=None , names= ["name_of_pic" , "noisy_label"])
		#print(selfdata.head())
		self.labels = data.noisy_label.tolist()
		# read addresses and labels from the 'train' folder
		self.pic_names = data.name_of_pic.tolist()
# 		self.weights = (collections.Counter(self.pic_names).values())
# 		print(self.weights)
		
# 		from numpy.random import choice
# 		l = [choice(self.pic_names, p=weights) for _ in range(100)]
# 		import collections
# 		# print("Length of val data is : ",len(val_data))
		
		self.root_dir = root_dir
		self.flag = training
		self.transform = transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir,self.pic_names[idx])
		

		image = ndimage.imread(img_name, mode="RGB")
		image = misc.imresize(image, (256,256),mode='RGB')
		#image = transform.resize(image , (128,128))
		# 
		#image = resizeimage.resize_cover(Image.open(img_name), [128, 128])
		
		if(self.flag == True):
			image = augment_images(image)
		

		correct_label = self.labels[idx]
		#correct_label = correct_label.astype('int')
		if self.transform:
			image = self.transform(image)
		
		
		return (image, correct_label)
	

	
	
	


def train(train_loader, model, save_location, training=False):
	"""Train for one epoch on the training set"""   
	
	
	print(torch.cuda.current_device())
	print(torch.cuda.device_count())
	print(torch.cuda.get_device_name(0))
	
	
	
	

	# switch to train mode
	model.eval()

	end = time.time()
	
	
	if(training):
		length_dataset = len(train_dataset)
		feat_holder = np.zeros([length_dataset,2048])
		label_holder = np.zeros([length_dataset])
	else:
		length_dataset = len(val_dataset)
		feat_holder = np.zeros([length_dataset,2048])
		label_holder = np.zeros([length_dataset])
		
	
# for num_ph,photo in enumerate(ph):
#     fp = './train_photos/'+str(photo)+'.jpg'
#     

	
	for i, (inp, target) in tqdm(enumerate(train_loader)):
		
		#print("love ")
		#print(i)
		target = target.type(torch.LongTensor).cuda(async=True)
		inp = inp.cuda()
		
		input_var = torch.autograd.Variable(inp)
		target_var = torch.autograd.Variable(target)
		#print(input_var.shape)
		# compute output
		
		with torch.no_grad():
			_, output = model(input_var)
			#weight = model.module.features.layer[-1].weight
			#print(output.data.shape)
			#print(target_var)
			if( length_dataset < args.batch_size * (i+1)  ):
				feat_holder[(i)* args.batch_size : , :]= output.squeeze().cpu().data.numpy()
				label_holder[(i)* args.batch_size : ] = target_var.cpu().data.numpy()

			else :
				feat_holder[(i)* args.batch_size : (i+1)*args.batch_size, :]= output.squeeze().cpu().data.numpy()
				label_holder[(i)* args.batch_size : (i+1)*args.batch_size] = target_var.cpu().data.numpy()
				#print(feat_holder.shape)
    
		#itt = input()
			
			
	
	d1={'features': feat_holder, 'labels':label_holder}
		
	
# 	np.save("d1.npy", d1)
# 	d2=np.load("d1.npy")
# 	print d1.get('key1')
# 	print d2.item().get('key2')
	
	np.save(save_location, d1)
					
	

		
		
		
		
		
		
		
		
		








load_model_path = "/home/mil/noguchi/M1/ifood/foodx/runs/food004/model_100.pth.tar"
best_model = torch.load( load_model_path)


model = best_model['model']





kwargs = {'num_workers': 4, 'pin_memory': True}
#assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')

train_data_path ="/home/mil/gupta/ifood18/data/train_set_clean/"
val_data_path ="/home/mil/gupta/ifood18/data/val_set/"

train_label ="/home/mil/gupta/ifood18/data/labels/train_info.csv"
val_label ="/home/mil/gupta/ifood18/data/labels/val_info.csv"

train_dataset = FoodDataset(train_data_path,train_label, transform_train, training= True)
val_dataset = FoodDataset(val_data_path, val_label, transform_test)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128 , shuffle = True, num_workers=4)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size= 128,num_workers=4 )




# get the number of model parameters
# print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# for training on multiple GPUs.
# Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
#model = torch.nn.DataParallel(model).cuda()
#model = model.cuda()




print(model)
# for epoch in range(args.start_epoch, args.epochs):
# 	adjust_learning_rate(optimizer, epoch+1)

# train for one epoch
train(train_loader, model, save_location = './data/xgboost_feat/train_3_times.npy', True)
# train for one epoch
train(val_loader, model, save_location = './data/xgboost_feat/val_times.npy', training = False)


	
# 		loss = criterion(output, target_var)

# 		prec3 = accuracy(output.data, target, topk=(1,3))[-1] ## res is of shape [2,B] representing top 1 and top k accuracy respectively
# 		losses.update(loss.data[0], inp.size(0))
# 		top3.update(prec3[0], inp.size(0))
		
	
# 		optimizer.zero_grad()
# 		loss.backward()
# 		optimizer.step()


# 		if i % args.print_freq == 0:

			
# 			# measure elapsed time
# 			batch_time.update(time.time() - end)
# 			end = time.time()
# 			print('Epoch: [{0}][{1}/{2}]\t'
# 				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
# 				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
# 				  'Prec@1 {top3.val:.3f} ({top3.avg:.3f})'.format(
# 					  epoch, i, len(train_loader), batch_time=batch_time,
# 					  loss=losses, top3=top3))
# 	# log to TensorBoard
# 	if args.tensorboard:
# 		log_value('train_loss', losses.avg, epoch)
# 		log_value('train_acc', top1.avg, epoch)

		



# '''
# Given a numpy array of probabilities, this function returns classification array (of either 0 or 1)
# The thresholding is performed using the parameter thresh
# if prob>thresh return 1, else return 0
# '''
# def np_thresh(x1,thresh):
    
#     x1 = x1-thresh
    
#     x1 = x1>0
    
#     x1 = np.array(x1,dtype=int)
    
#     return x1
    
# '''
# Utility function to calculate precision,recall,fscore
# '''
# def fscore(x1,x2):

#     tru_pos  = np.sum(np.array((np.logical_and(x1,x2)),dtype=int))

#     actual_pos = np.sum(x1)
    
#     pred_pos = np.sum(x2)
#     if(pred_pos>0):
#         precision = float(tru_pos)/pred_pos
#     else:
#         precision = 0
    
#     if(actual_pos>0):
#         recall = float(tru_pos)/actual_pos
#     else:
#         recall=0
    
    
#     if(precision+recall>0):
#         fscore = (2*precision*recall)/(precision+recall)
#     else:
#         fscore = 0
    
      
    
#     return precision,recall,fscore

# '''
# train.csv in this problem is given such that for each business, we have a corresponding string entry 
# in the format '2 3 4 5', representing which labels are positive for that business

# This function clean_train creates a new csv file where each entry has nine fields.

# For example an entry '2 3 4 5' is coded as 0 0 1 1 1 1 0 0 0 

# The resulting file is saved as train_cl.csv

# The main program will almost always use train_cl.csv
# '''
# def clean_train():
#     train = pd.read_csv('train.csv')  
    
#     for i in range(9):
#         col_name = 'label_'+str(i)
        
#         train[col_name] = train['labels'].apply(lambda x: 1 if (str(i) in str(x)) else 0)
    
#     train = train.drop(['labels'],axis=1)
    
#     train.to_csv('train_cl.csv',index=0)

start_time = time.time()

def accuracy_score_new(y_pred, y_true):
	matched = 0
	x = y_true.astype(int)
	print(x.shape)
	print(x[0])
	for y_p, y_t in zip(y_pred, x):
		print(y_t , y_p)
		if y_t in y_p:
			matched = matched + 1

	dum = len(y_true) * 1.0
	return (matched / dum)





d2=np.load('./data/xgboost_feat/train.npy')
# 	print d1.get('key1')
train_features = d2.item().get('features')
train_labels = d2.item().get('labels')


d1=np.load('./data/xgboost_feat/val.npy')
# 	print d1.get('key1')
val_features = d1.item().get('features')
val_labels = d1.item().get('labels')


#train = data[:int(sz[0] * 0.7), :]
#test = data[int(sz[0] * 0.7):, :]

train_X = train_features
train_Y = train_labels

test_X = val_features
test_Y = val_labels

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost








# xgb1 = XGBClassifier(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=1,
#  gamma=0,
#  gpu_id= 0,
#  tree_method = 'gpu_hist',
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'multi:softprob',
#  num_class= args.class_size,
#  nthread=8,
#  scale_pos_weight=1,
#  seed=27)












param = {}
# use softmax multi-class classification
#param['objective'] = 'multi:softmax'
# scale weight of positive examples
# param = {}
param['objective'] = 'multi:softprob'
# #param['objective'] = 'multi:softmax'
param['eta'] = 0.1
param['max_depth'] = 3
param['subsample'] = 0.6
#param['silent'] = 1
param['nthread'] = 16
param['gpu_id'] = 0
param['max_bin'] = 256
param['tree_method'] = "gpu_hist"
param['gamma']=0
param['eval_metric'] = "mlogloss"
num_round = 500
param['num_class'] = args.class_size


watchlist = [ (xg_train, 'train'), (xg_test, 'eval') ]
#print(xg_test.shape)
#num_round = 5
#bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=5)

#bst.save_model("/home/mil/gupta/ifood18/data/xgboost_feat/xgboost_model_saved_max_3_depth.dat")

bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model("/home/mil/gupta/ifood18/data/xgboost_feat/xgboost_model_saved_max_3_depth.dat")
# get prediction
# pred = bst.predict_proba(xg_test)



topk = 3
probs = bst.predict(xg_test)
print(probs.shape)
best_n = np.argsort(probs, axis=1)[:,-topk:]
print(best_n.shape)
print(test_Y.shape)
print(type(test_Y))
ans = accuracy_score_new( best_n, test_Y)
print(ans)


#error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
#print('Test error using softmax = {}'.format(error_rate))

print("Took time ", time.time()- start_time)





# def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
#     if useTrainCV:
#         xgb_param = alg.get_xgb_params()
#         xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
#         cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
#             metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
#         alg.set_params(n_estimators=cvresult.shape[0])
    
#     #Fit the algorithm on the data
#     alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
        
#     #Predict training set:
#     dtrain_predictions = alg.predict(dtrain[predictors])
#     dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
#     #Print model report:
#     print "\nModel Report"
#     print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
#     print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
                    
#     feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
#     feat_imp.plot(kind='bar', title='Feature Importances')
#     plt.ylabel('Feature Importance Score')









	
	
# # some time later...
 
# # load model from file
# loaded_model = joblib.load("pima.joblib.dat")
# # make predictions for test data
# y_pred = loaded_model.predict(X_test)



# # do the same thing again, but output probabilities
# param['objective'] = 'multi:softprob'
# bst = xgb.train(param, xg_train, num_round, watchlist)
# # Note: this convention has been changed since xgboost-unity
# # get prediction, this is in 1D array, need reshape to (ndata, nclass)
# pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 6)
# pred_label = np.argmax(pred_prob, axis=1)
# error_rate = np.sum(pred_label != test_Y) / test_Y.shape[0]
# print('Test error using softprob = {}'.format(error_rate))






# d1={'features': feat_holder, 'labels':label_holder}


# 	np.save("d1.npy", d1)
# d2=np.load('./data/xgboost_feat/xgboost_holder.npy',mmap_mode='r')
# # 	print d1.get('key1')
# features = d2.item().get('features')
# labels = d2.item().get('labels')
# np.save('./data/xgboost_feat/xgboost_holder.npy', d1)



# '''
# Part1:Cleaning

# After pretraining, we have two files feat_holder.npy which is a num_unique_train_photosx2048 numpy array

# and feat_holder_test.npy which is a num_unique_test_photosx2048 numpy array

# Since each business has multiple instances (photos), we need some way to

# condense this multiple instance information to a single instance information

# The simplest way to do this is to take the mean of all the instances per business, and assign it

# as a feature vector for that business.

# At the end of this, we have a num_unique_businessx2048 array which we will use as a festure vector for training

# The result of this part of code is a pandas dataframe 

# '''

# train_to_biz = pd.read_csv('train_photo_to_biz_ids.csv')

# train_image_features = np.load('feat_holder.npy',mmap_mode='r')

# uni_bus = train_to_biz['business_id'].unique()

# coll_arr = np.zeros([len(uni_bus),2048])

# for nb,ub in enumerate(uni_bus):
#     tbz = np.array(train_to_biz['business_id']==ub,dtype=bool)
#     x1 = np.array(train_image_features[tbz,:])
#     x1 = np.mean(x1,axis=0)
#     x1 = x1.reshape([1,2048])
#     coll_arr[nb,:2048] = x1

    
# biz_features = pd.DataFrame(uni_bus,columns=['business_id'])

# coll_arr = pd.DataFrame(coll_arr)

# frames = [biz_features,coll_arr]

# biz_features = pd.concat(frames,axis=1)

# del train_to_biz,train_image_features,coll_arr,frames

# print('time taken for Part1:Cleaning',time.time()-start_time)



# '''
# Part2: Estimating the performance using cross validation

# We will use 5 fold cross validation to estimate the performance of our model,

# and in the process tune the parameters of the model

# Here we will be using the xgboost package

# This part of code will also serve a second purpose

# If we would like to use this model in an ensemble at a later stage, we will save the results

# from cross validation, and use them as features in the ensembling stage

# Since xgboost doesnt have a straightforward way of training multi label classification problems,

# we will build 9 different binary classification models. However, this does not take into account the 

# realtionship between the different labels, and might not result in the best performance

# We will overcome this deficiency in the ensembling stage by using a slightly different architecture

# To calculate the fscore, we will need to use some threshold to convert the probabilites into binary labels

# The thresholding step is not very critical if our goal is to use this model only as features for the second level ensemble

# But if this model is the final model, then the thresholding parameter also needs to be tuned.

# Here, I've used 0.48 as the threshold


# '''


# cv = 1

# submit = 0

# num_cv = 5

# from sklearn.cross_validation import KFold

# skf = list(KFold(biz_features.shape[0],num_cv,random_state=42))

# dataset_blend_train = np.zeros([biz_features.shape[0],9])

# #labels = ['label_'+str(i) for i in range(9)]

# param = {}
# fparam['objective'] = 'multi:softprob'
# #param['objective'] = 'multi:softmax'
# param['eta'] = 0.1
# param['max_depth'] = 3
# param['subsample'] = 0.6
# param['silent'] = 1
# param['nthread'] = 4
# param['eval_metric'] = "mlogloss"
# num_round = 100
# param['num_class'] = args.num_classes


# iter_label = {}

# for nb,lb in enumerate(labels):
#     iter_n = 0
#     train_cl = pd.read_csv('train_cl.csv')

#     train_cl = dict(np.array(train_cl[['business_id',lb]]))

#     biz_features['lb'] = biz_features['business_id'].apply(lambda x: train_cl[x])
#     if(cv):
#         for (training,testing) in skf:
        
#             df_train = biz_features.iloc[training]
            
#             df_test = biz_features.iloc[testing]

#             df_train_values = np.array(df_train['lb'],dtype=bool)
            
#             df_train_features = df_train.drop(['business_id','lb'],axis=1)
            
#             df_test_values = np.array(df_test['lb'],dtype=bool)
            
#             df_test_features = df_test.drop(['business_id','lb'],axis=1)

#             xg_train = xgb.DMatrix(df_train_features, label=df_train_values)
            
#             xg_test = xgb.DMatrix(df_test_features, label=df_test_values)
            
#             watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
            
#             bst = xgb.train(param, xg_train, num_round,watchlist,early_stopping_rounds=4,verbose_eval=0)
            
#             iter_n = iter_n+ bst.best_iteration
            
#             yprob = bst.predict(xg_test)[:,1]
            
#             dataset_blend_train[testing,nb] = yprob

#         iter_label[lb] = int(float(iter_n)/num_cv)

# # calculate precision

# np.save('./Stack/Model1_Full.npy',dataset_blend_train)

# dataset_blend_train2 = np_thresh(dataset_blend_train,0.48)

# fsc = np.zeros(dataset_blend_train2.shape[0])

# for nb,lb in enumerate(labels):
#     train_cl = pd.read_csv('train_cl.csv')

#     train_cl = dict(np.array(train_cl[['business_id',lb]]))

#     biz_features[lb] = biz_features['business_id'].apply(lambda x: train_cl[x])

# truth = np.array(biz_features[labels])

# for i in range(dataset_blend_train2.shape[0]):
    
#     prec,recall,fsc[i] = fscore(truth[i,:],dataset_blend_train2[i,:])

# print('Fscore on this model:',np.mean(fsc))

# print('time taken for cross validation',time.time()-start_time)




'''
Part3: Training and Generating submissions on the Test Set

Now that we have tuned the model parameters using cross validation, we will go ahead

and use these parameters to build 9 binary classification models

Set the submit variable to 1 only if you need to run this part, since in most cases we will only be

playing with the cross validation part


Also, we intend to use this model in an ensemble, we will store the predictions on the test

set, and use them later as features for the ensemble model

'''

# if(submit):
#     param['subsample'] = param['subsample']*0.8
#     train_to_biz = pd.read_csv('train_photo_to_biz_ids.csv')

#     train_image_features = np.load('feat_holder.npy',mmap_mode='r')
    
#     uni_bus = train_to_biz['business_id'].unique()
    
#     coll_arr = np.zeros([len(uni_bus),2048])
    
#     for nb,ub in enumerate(uni_bus):
#         if(nb%1000==0):
#             print(nb)
#         tbz = np.array(train_to_biz['business_id']==ub,dtype=bool)
#         x1 = np.array(train_image_features[tbz,:])
#         x1 = np.mean(x1,axis=0)
#         x1 = x1.reshape([1,2048])
#         coll_arr[nb,:] = x1
        
#     biz_features = pd.DataFrame(uni_bus,columns=['business_id'])
    
#     coll_arr = pd.DataFrame(coll_arr)
    
#     frames = [biz_features,coll_arr]
    
#     biz_features = pd.concat(frames,axis=1)
    
#     del train_to_biz,train_image_features,coll_arr,frames
    
#     model_dict = {}
    
#     for nb,lb in enumerate(labels):
#         train_cl = pd.read_csv('train_cl.csv')
    
#         train_cl = dict(np.array(train_cl[['business_id',lb]]))
    
#         biz_features['lb'] = biz_features['business_id'].apply(lambda x: train_cl[x])
            
        
#         df_train_values = biz_features['lb']
        
#         df_train_features = biz_features.drop(['business_id','lb'],axis=1)
        
#         xg_train = xgb.DMatrix(df_train_features, label=df_train_values)

#         bst = xgb.train(param, xg_train,iter_label[lb])

#         model_dict[lb] = bst
        
#         df_train_features = None
        
#         df_test_features = None
        
#         xg_train = None
    
#     # Predict on the test set
    
#     test_to_biz = pd.read_csv('test_photo_to_biz.csv')

#     test_image_features = np.load('feat_holder_test.npy',mmap_mode='r')
     
#     test_image_id = list(np.array(test_to_biz['photo_id'].unique()))
     
#     uni_bus = test_to_biz['business_id'].unique()
     
#     coll_arr = np.zeros([len(uni_bus),2048])
     
#     for nb,ub in enumerate(uni_bus):
#         if(nb%1000==0):
#             print(nb)
#         image_ids = test_to_biz[test_to_biz['business_id']==ub]['photo_id'].tolist()  
#         image_index = [test_image_id.index(x) for x in image_ids]
#         features = test_image_features[image_index]
#         x1 = np.mean(features,axis=0)
#         x1 = x1.reshape([1,2048])
#         coll_arr[nb,:] = x1

        
#     biz_features = pd.DataFrame(uni_bus,columns=['business_id'])
    
#     coll_arr = pd.DataFrame(coll_arr)
    
#     frames = [biz_features,coll_arr]
    
#     biz_features = pd.concat(frames,axis=1)
    
#     del coll_arr,frames,test_to_biz,test_image_features,test_image_id,image_ids,image_index,features
    
#     result = np.zeros([biz_features.shape[0],9])
    
#     for nb,lb in enumerate(labels):
        
#         print('predicting',lb)
#         df_test_features = biz_features.drop(['business_id'],axis=1)
        
#         bst = model_dict[lb]
        
#         yprob = bst.predict(xgb.DMatrix(df_test_features))[:,1]
        
#         result[:,nb] = yprob
    
#     np.save('./Stack/Model1_Full_result.npy',result)
    
#     result = np_thresh(result,0.480)
    
#     bid = np.array(biz_features['business_id'])
    
#     fin = {}
    
#     for i in range(result.shape[0]):
#         x = result[i,:]
#         li = [((q)) for q in range(9) if x[q]==1]
#         fin[bid[i]] = li
        
#     for j in fin.keys():
#         fin[j] = ' '.join(str(e) for e in fin[j])
    
#     x1 = pd.DataFrame(biz_features['business_id'])
    
#     x1['labels'] = x1['business_id'].apply(lambda x: fin[x] if x in fin.keys() else '0')
    
#     x1.to_csv('result.csv',index=0)





    
    