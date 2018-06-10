

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


parser.add_argument('--batch-size', default=128, type=int,
					help='batch size for inferencing (default: 128)')

parser.add_argument('--class-size', default=211, type=int,
					help='class size depending upon dataset  (default: 211 for ifood dataset)')


parser.add_argument('--resize-size', default=256, type=int,
					help='resize image size depending upon dataset  (default: 256 for ifood dataset)')

parser.add_argument('--crop-size', default=224, type=int,
					help='size of image to be input for the model  (default: 224 for resnet 152 model)')




args = parser.parse_args()


def accuracy_score_new(y_pred, y_true):
	matched = 0
	for y_p, y_t in zip(y_pred, y_true):
		if y_t in y_p:
			matched = matched + 1

	dum = len(y_true) * 1.0
	return (matched / dum)


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
param['silent'] = 1
param['nthread'] = 16
param['gpu_id'] = 0
param['max_bin'] = 256
param['tree_method'] = "gpu_hist"
param['gamma']=0
param['eval_metric'] = "mlogloss"
num_round = 200
param['num_class'] = args.class_size


#watchlist = [(xg_train, 'train'), (xg_test, 'test')]
#num_round = 5
#bst = xgb.train(param, xg_train, num_round, watchlist)
bst = joblib.load( "/home/mil/gupta/ifood18/data/xgboost_feat/xgboost_model_saved.dat")	
# get prediction
# pred = bst.predict(xg_test)
print(type(bst))
clf = XGBClassifier()
booster = Booster()
booster.load_model('./model.xgb')
clf._Booster = booster


topk = 3
probs = .predict_proba(xg_test)
best_n = np.argsort(probs, axis=1)[-topk:]
print(best_n)
print(best_n.shape)
itt = input()
accuracy_score_new(y_true, y_pred)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))

print("Took time ", time.time()- start_time)

