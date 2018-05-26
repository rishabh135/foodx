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





test_data_path ="/home/mil/gupta/ifood18/data/test_set/"
test_label_path ="/home/mil/gupta/ifood18/data/labels/test_info.csv"


data = pandas.read_csv(test_label_path, header=None , names= ["name_of_pic"])
print(data.head())
#If you want your lists as in the question, you can now do:


#val_labels = data.noisy_label.tolist()
# read addresses and labels from the 'train' folder
test_data = data.name_of_pic.tolist()
print("Length of val data is : ",len(test_data))
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
	test_shape = (len(test_data),3, image_size, image_size)
# 	val_shape = (len(val_data_path), image_size, image_size, 3)
# 	test_shape = (len(test_data_path), image_size, image_size, 3)


	
	


hdf5_path = "/home/mil/gupta/ifood18/data/h5data/test_data.h5py"
# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')

hdf5_file.create_dataset("data", test_shape, np.float32)
hdf5_file.create_dataset("mean", test_shape[1:], np.float32)
#hdf5_file.create_dataset("labels", (len(test_data),), np.int32)
#hdf5_file["labels"][...] = val_labels



from tqdm import tqdm
# a numpy array to save the mean of the images
mean = np.zeros(test_shape[1:], np.float32)
# loop over train addresses
for i in tqdm(range(len(test_data))):			   
	addr = os.path.join(test_data_path,test_data[i])
	#print("image addres is :",addr)
	img = cv2.imread(addr)
	#print(img)
	img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	


	#current_image = ia.imresize_single_image(addr, (256, 256))
	image_aug = img.transpose(2,0,1)
	hdf5_file["data"][i, ...] = image_aug[None]
	mean += image_aug / float(len(test_data))

	
	
	
print("time taken for image" , time.time()-start)
hdf5_file["mean"][...] = mean
hdf5_file.close()