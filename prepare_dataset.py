import numpy as np, os
import h5py
import glob
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from time import time




image_size = 256






train_data_path ="/home/mil/gupta/ifood18/data/train_set/"
val_data_path ="/home/mil/gupta/ifood18/data/val_set/"
test_data_path ="/home/mil/gupta/ifood18/data/test_set/"
start =time.time()
data_order = 'pytorch'# 'th' for Theano, 'tf' for Tensorflow
# check the order of data and chose proper data shape to save images
# if data_order == 'th':
#     train_shape = (len(train_addrs), 3,)
#     val_shape = (len(val_addrs), 3, 224, 224)
#     test_shape = (len(test_addrs), 3, 224, 224)
# elif 
if data_order == 'pytorch':
	train_shape = (len(train_data_path), image_size, image_size, 3)
	val_shape = (len(val_data_path), image_size, image_size, 3)
	test_shape = (len(test_data_path), image_size, image_size, 3)



hdf5_path = "/home/mil/gupta/ifood18/data/h5data/"
# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')

hdf5_file.create_dataset("train_img", train_shape, np.int8)
hdf5_file.create_dataset("val_img", val_shape, np.int8)
hdf5_file.create_dataset("test_img", test_shape, np.int8)
hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
hdf5_file.create_dataset("train_labels", (len(train_data_path),), np.int8)
hdf5_file["train_labels"][...] = train_labels
hdf5_file.create_dataset("val_labels", (len(val_data_path),), np.int8)
hdf5_file["val_labels"][...] = val_labels
hdf5_file.create_dataset("test_labels", (len(test_data_path),), np.int8)
hdf5_file["test_labels"][...] = test_labels




def findFilesInFolder(path, pathList, extension, subFolders = True):
    """  Recursive function to find all files of an extension type in a folder (and optionally in all subfolders too)

    path:        Base directory to find files
    pathList:    A list that stores all paths
    extension:   File extension to find
    subFolders:  Bool.  If True, find files in all subfolders under path. If False, only searches files in the specified folder
    """

    try:   # Trapping a OSError:  File permissions problem I believe
        for entry in os.scandir(path):
            if entry.is_file() and entry.path.endswith(extension):
                pathList.append(entry.path)
            elif entry.is_dir() and subFolders:   # if its a directory, then repeat process as a nested function
                pathList = findFilesInFolder(entry.path, pathList, extension, subFolders)
    except OSError:
        print('Cannot access ' + path +'. Probably a permissions error')

    return pathList


extension = ".jpg"

train_data = []
train_data = findFilesInFolder(train_data_path, train_data, extension, True)



print(train_data)

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values



# a numpy array to save the mean of the images
mean = np.zeros(train_shape[1:], np.float32)
# loop over train addresses
for i in range(len(train_data_path)):
	# print how many images are saved every 1000 images
	# if( i % 1000 == 0 and i > 1):
	#	print ('Train data: {}/{}'.format(i, len(train_addrs))
	# read an image and resize to (224, 224)
	# cv2 load images as BGR, convert it to RGB
	
			   
			   
	addr = train_data[i]

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

	current_image = ia.imresize_single_image(addr, (256, 256))
	images_aug = seq.augment_images(current_image)
	
	
#     img = cv2.imread(addr)
#     img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# add any image pre-processing here
	# if the data order is Theano, axis orders should change
	# if data_order == 'th':
	#     img = np.rollaxis(img, 2)
	# save the image and calculate the mean so far
	hdf5_file["train_img"][i, ...] = images_aug
	mean += current_image / float(len(train_labels))

	print("time taken for image" , time.time()-start)
	start = time.time()
	# loop over validation addresses


# for i in range(len(val_addrs)):
#     # print how many images are saved every 1000 images
#     if i % 1000 == 0 and i > 1:
#         print 'Validation data: {}/{}'.format(i, len(val_addrs))
#     # read an image and resize to (224, 224)
#     # cv2 load images as BGR, convert it to RGB
#     addr = val_addrs[i]
#     img = cv2.imread(addr)
#     img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # add any image pre-processing here
#     # if the data order is Theano, axis orders should change
#     if data_order == 'th':
#         img = np.rollaxis(img, 2)
#     # save the image
#     hdf5_file["val_img"][i, ...] = img[None]
# # loop over test addresses
# for i in range(len(test_addrs)):
#     # print how many images are saved every 1000 images
#     if i % 1000 == 0 and i > 1:
#         print 'Test data: {}/{}'.format(i, len(test_addrs))
#     # read an image and resize to (224, 224)
#     # cv2 load images as BGR, convert it to RGB
#     addr = test_addrs[i]
#     img = cv2.imread(addr)
#     img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # add any image pre-processing here
#     # if the data order is Theano, axis orders should change
#     if data_order == 'th':
#         img = np.rollaxis(img, 2)
#     # save the image
#     hdf5_file["test_img"][i, ...] = img[None]
# # save the mean and close the hdf5 file
hdf5_file["train_mean"][...] = mean
hdf5_file.close()