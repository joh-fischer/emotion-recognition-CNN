import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.model_selection import train_test_split


def load_RAF_targetvector(folderpath, target_img_size=(100, 100), preprocessing_function=None, batch_size=32):
	"""
	Load RAF data with target vector being a probability distribution out of the votes.
	
	Args:
		folderpath (str): Path where the data is stored with the following structure
							/distribution_basic.txt
							/images
								/test_0001_aligned.jpg
								/test_0002_aligned.jpg
								/test_0003_aligned.jpg
								/...
		target_img_size (tuple): Target size of the image, default (100, 100).
		preprocessing_function (function): Function applied to each image as preprocessing step, default None.
		batch_size (int): The number of samples in a batch, default 32.
	Returns:
		train_data, val_data, test_data (ImageDataGenerators): Training, validation and test data generators.
	"""
	# get file and folder paths
	labelling_list = open(folderpath + 'distribution_basic.txt', 'r').read().strip().split(' \n')
	img_dir = folderpath + 'images/'

	X_train = []; Y_train = []
	X_test = []; Y_test = []
	Y_test_oneclass = []
	
	for name_vector_string in labelling_list:
		# separate line of labelling list into image name and target vector
		splitted_name_vector_string = name_vector_string.split(' ')
		image_name = splitted_name_vector_string[0]
		# typecast target vector from string to float
		target_vector = np.asarray(splitted_name_vector_string[1:], dtype="float32")
		
		# get class label to stratify training / validation split
		max_label = np.argmax(target_vector)

		# add aligned to image
		name, fileend = image_name.split('.')
		filename = name + '_aligned.' + fileend

		img = cv2.imread(img_dir + filename)
		if img is None:
			print("Error finding image", filename)
			continue
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# resize image
		if img.shape[:2] != target_img_size:
			img = cv2.resize(img, dsize=target_img_size, interpolation=cv2.INTER_CUBIC)

		if filename.startswith('train'):
			X_train.append(img)
			Y_train.append(target_vector)
		else:
			X_test.append(img)
			Y_test.append(target_vector)
			Y_test_oneclass.append(max_label)

	X_train = np.array(X_train); Y_train = np.array(Y_train)
	X_test = np.array(X_test); Y_test = np.array(Y_test)

	print("Splitting test dataset into stratified validation and test set")
	X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, stratify=Y_test_oneclass)


	print("Training\n-", X_train.shape, "\n-", Y_train.shape)
	print("Validation\n-", X_val.shape, "\n-", Y_val.shape)
	print("Testing\n-", X_test.shape, "\n-", Y_test.shape)

	train_data_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
	                                    height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
	                                    horizontal_flip=True,
	                                    preprocessing_function=preprocessing_function)
	train_data = train_data_gen.flow(
		x=X_train, y=Y_train,
		batch_size=batch_size,
		shuffle=True)

	val_data_gen = ImageDataGenerator(preprocessing_function=preprocessing_function)
	val_data = val_data_gen.flow(
		x=X_val, y=Y_val,
		batch_size=batch_size,
		shuffle=True)

	test_data_gen = ImageDataGenerator(preprocessing_function=preprocessing_function)
	test_data = test_data_gen.flow(
		x=X_test, y=Y_test,
		batch_size=batch_size,
		shuffle=True)

	return train_data, val_data, test_data

    
def load_FERplus_targetvector(folderpath, target_img_size=(100, 100), preprocessing_function=None, batch_size=32):
	"""
	Load FER+ data with target vector being a probability distribution of the votes.
	
	Args:
		folderpath (str): Path where the data is stored with the following structure
							/fer2013new.csv
							/images
								/FER2013Test
									/fer0032220.png
									/...
								/FER2013Train
									/fer0000000.png
									/...
								/FER2013Valid
									/fer0028638.png
									/...
		target_img_size (tuple): Target size of the image, default (100, 100).
		preprocessing_function (function): Function applied to each image as preprocessing step, default None.
		batch_size (int): The number of samples in a batch, default 32.
	Returns:
		train_data, val_data, test_data (ImageDataGenerators): Training, validation and test data generators.
	"""
	path_fer2013new = folderpath + 'fer2013new.csv'
	train_dir = folderpath + 'images/FER2013Train/'
	test_dir = folderpath + 'images/FER2013Test/'
	val_dir = folderpath + 'images/FER2013Valid/'

	labelling_list = pd.read_csv(path_fer2013new)

	X_train = []; Y_train = []
	X_val = []; Y_val = []
	X_test = []; Y_test = []

	# iterate through files and load them
	for idx, elem in labelling_list.iterrows():
		image_name = elem['Image name']
	    
		if not isinstance(image_name, str): continue
			
		# get the vote vector for the emotions
		vote_vector = np.array(elem[2:-3].to_numpy(), dtype='float32')
		
		# skip if sum of vote vector is smaller than 1 (less than one vote for image)
		if np.sum(vote_vector) < 1: continue
		
		# transform to probability distribution
		prob_dist_vote_vector = vote_vector / np.sum(vote_vector)
		
		if elem.Usage == 'Training':
			img = cv2.imread(train_dir + image_name)
			if img is None: continue
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			if img.shape[:2] != target_img_size:
				img = cv2.resize(img, dsize=target_img_size, interpolation=cv2.INTER_CUBIC)

			X_train.append(img)
			Y_train.append(prob_dist_vote_vector)
			
		elif elem.Usage == 'PrivateTest':
			img = cv2.imread(test_dir + image_name)
			if img is None: continue
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			if img.shape[:2] != target_img_size:
				img = cv2.resize(img, dsize=target_img_size, interpolation=cv2.INTER_CUBIC)

			X_val.append(img)
			Y_val.append(prob_dist_vote_vector)
			
		elif elem.Usage == 'PublicTest':
			img = cv2.imread(val_dir + image_name)
			if img is None: continue
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			if img.shape[:2] != target_img_size:
				img = cv2.resize(img, dsize=target_img_size, interpolation=cv2.INTER_CUBIC)

			X_test.append(img)
			Y_test.append(prob_dist_vote_vector)
	        
	X_train = np.array(X_train); Y_train = np.array(Y_train)
	X_val = np.array(X_val); Y_val = np.array(Y_val)
	X_test = np.array(X_test); Y_test = np.array(Y_test)

	print("Training\n-", X_train.shape, "\n-", Y_train.shape)
	print("Validation\n-", X_val.shape, "\n-", Y_val.shape)
	print("Testing\n-", X_test.shape, "\n-", Y_test.shape)

	train_data_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
	                                    height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
	                                    horizontal_flip=True,
	                                    preprocessing_function=preprocessing_function)
	train_data = train_data_gen.flow(
		x=X_train, y=Y_train,
		batch_size=batch_size,
		shuffle=True)

	val_data_gen = ImageDataGenerator(preprocessing_function=preprocessing_function)
	val_data = val_data_gen.flow(
		x=X_val, y=Y_val,
		batch_size=batch_size,
		shuffle=True)

	test_data_gen = ImageDataGenerator(preprocessing_function=preprocessing_function)
	test_data = test_data_gen.flow(
		x=X_test, y=Y_test,
		batch_size=batch_size,
		shuffle=True)

	return train_data, val_data, test_data


if __name__ == "__main__":
	print("\nRAF:")
	RAF_DIR = './data/RAF/'
	load_RAF_targetvector(RAF_DIR)

	print("\nFER+:")
	FER_DIR = './data/ferplus2013/'
	load_FERplus_targetvector(FER_DIR)