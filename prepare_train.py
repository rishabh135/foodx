import numpy as np, os
import h5py
import glob
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import time, csv
import cv2
import pandas


image_size = 256






train_data_path ="/home/mil/gupta/ifood18/data/train_set/"
train_label_path ="/home/mil/gupta/ifood18/data/labels/train_info.csv"


data = pandas.read_csv(train_label_path, header=None , names= ["name_of_pic" , "noisy_label"])
print(data.head())
#If you want your lists as in the question, you can now do:


train_labels = data.noisy_label.tolist()
train_data = data.name_of_pic.tolist()
print("Length of train data is : ",len(train_data))
#combined = zip(train_data, train_labels)





start =time.time()
data_order = 'pytorch'# 'th' for Theano, 'tf' for Tensorflow
# check the order of data and chose proper data shape to save images
# if data_order == 'th':
#     train_shape = (len(train_addrs), 3,)
#     val_shape = (len(val_addrs), 3, 224, 224)
#     test_shape = (len(test_addrs), 3, 224, 224)
# elif 
if data_order == 'pytorch':
	train_shape = (len(train_data),3, image_size, image_size)
# 	val_shape = (len(val_data_path), image_size, image_size, 3)
# 	test_shape = (len(test_data_path), image_size, image_size, 3)


	
	


hdf5_path = "/home/mil/gupta/ifood18/data/h5data/train_data_iteration_1.h5py"
# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("data", train_shape, np.float32)
hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
hdf5_file.create_dataset("labels", (len(train_data),), np.int32)
hdf5_file["labels"][...] = train_labels





# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
# add a random value from the range (-30, 30) to the first two channels of
# input images (e.g. to the R and G channels)
seq = iaa.Sequential([
	# apply the following augmenters to most images
	iaa.Fliplr(0.5), # horizontally flip 50% of all images
	iaa.Flipud(0.2), # vertically flip 20% of all images
	# crop images by -5% to 10% of their height/width
	sometimes(iaa.CropAndPad(
		percent=(-0.05, 0.1),
		pad_mode=ia.ALL,
		pad_cval=(0, 255)
	)),
	sometimes(iaa.Affine(
		scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
		translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
		rotate=(-45, 45), # rotate by -45 to +45 degrees
		shear=(-16, 16), # shear by -16 to +16 degrees
		order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
		cval=(0, 255), # if mode is constant, use a cval between 0 and 255
		mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
	))
])
# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values


from tqdm import tqdm
# a numpy array to save the mean of the images
mean = np.zeros(train_shape[1:], np.float32)
# loop over train addresses
for i in tqdm(range(len(train_data))):
		   
			   
	addr = os.path.join(train_data_path,train_data[i])
	#print("image addres is :",addr)
	img = cv2.imread(addr)
	#print(img)
	img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	


	#current_image = ia.imresize_single_image(addr, (256, 256))
	image_aug = seq.augment_image(img).transpose(2,0,1)
	hdf5_file["data"][i, ...] = image_aug[None]
	mean += image_aug / float(len(train_labels))


print("time taken for image" , time.time()-start)
start = time.time()
hdf5_file["mean"][...] = mean
hdf5_file.close()