# 2019 10 05 
# For generate MNIST data from the original 28*28 data
# With possible downsampling
# Binary output for specified two number in 0~9


from keras.datasets import mnist
import numpy as np 
import torch
import matplotlib.pyplot as plt

###

def plot_two_classes(data_set_category_1, data_set_category_2):

	x = np.arange(1024)

	fig, ax = plt.subplots(3,1)
	class_1 = ax[0].bar(x, data_set_category_1, color='b')
	class_2 = ax[0].bar(x, data_set_category_2, color='r')

	class_a = ax[1].bar(x, data_set_category_1, color='b')
	class_b = ax[2].bar(x, data_set_category_2, color='r')

	plt.show()
	
	return

def display_img(image):
	# original = np.transpose(original, (1, 2, 0))
	# adversarial = np.transpose(adversarial, (1, 2, 0))

	plt.figure()

	plt.title('Image')
	plt.imshow(image)  # division by 255 to convert [0, 255] to [0, 1]
	plt.axis('off')

	plt.show()

## Data Preprocessing ##


## Data Output ##

def torch_data_loading():
	pass

def down_sampling():
	pass

def get_target_num():
	pass


def data_loading_down_sampled():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	# Normalized
	x_train = x_train[:,5:23,5:23]
	x_test = x_test[:,5:23,5:23]

	# x_train = (x_train.reshape(60000, 784)/255. - 0.1307)/0.3081
	# x_test = (x_test.reshape(10000, 784)/255. - 0.1307)/0.3081

	x_train = (x_train.reshape(60000, 324)/255. - 0.1307)/0.3081
	x_test = (x_test.reshape(10000, 324)/255. - 0.1307)/0.3081

	x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_test).type(torch.FloatTensor)

	y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_test).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test

def data_loading():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	# Normalized

	x_train = (x_train.reshape(60000, 784)/255. - 0.1307)/0.3081
	x_test = (x_test.reshape(10000, 784)/255. - 0.1307)/0.3081

	x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_test).type(torch.FloatTensor)

	y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_test).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test

def data_loading_with_padding():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
	x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')
	# Normalized

	x_train = (x_train.reshape(60000, 1024)/255. - 0.1307)/0.3081
	x_test = (x_test.reshape(10000, 1024)/255. - 0.1307)/0.3081

	x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_test).type(torch.FloatTensor)

	y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_test).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test

def data_loading_with_target(target_num):
	# Should clean out the test data set also!!!!!!

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Normalized

	x_train = (x_train.reshape(60000, 784)/255. - 0.1307)/0.3081
	x_test = (x_test.reshape(10000, 784)/255. - 0.1307)/0.3081

	# Select out the train target

	x_true_train = []
	y_true_train = []

	for idx in range(len(y_train)):
		if y_train[idx] == target_num:
			x_true_train.append(x_train[idx])
			y_true_train.append(y_train[idx])

	x_true_train = np.array(x_true_train)
	y_true_train = np.array(y_true_train)

	# Select out the test target

	x_true_test = []
	y_true_test = []

	for idx in range(len(y_test)):
		if y_test[idx] == target_num:
			x_true_test.append(x_test[idx])
			y_true_test.append(y_test[idx])

	x_true_train = np.array(x_true_train)
	y_true_train = np.array(y_true_train)

	x_true_test = np.array(x_true_test)
	y_true_test = np.array(y_true_test)

	# x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_train = torch.from_numpy(x_true_train).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_true_test).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_train = torch.from_numpy(y_true_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_true_test).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test

def data_loading_with_padding_target(target_num):
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
	x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')

	# Normalized

	x_train = (x_train.reshape(60000, 1024)/255. - 0.1307)/0.3081
	x_test = (x_test.reshape(10000, 1024)/255. - 0.1307)/0.3081

	# Select out the train target

	x_true_train = []
	y_true_train = []

	for idx in range(len(y_train)):
		if y_train[idx] == target_num:
			x_true_train.append(x_train[idx])
			y_true_train.append(y_train[idx])

	x_true_train = np.array(x_true_train)
	y_true_train = np.array(y_true_train)

	# Select out the test target

	x_true_test = []
	y_true_test = []

	for idx in range(len(y_test)):
		if y_test[idx] == target_num:
			x_true_test.append(x_test[idx])
			y_true_test.append(y_test[idx])

	x_true_train = np.array(x_true_train)
	y_true_train = np.array(y_true_train)

	x_true_test = np.array(x_true_test)
	y_true_test = np.array(y_true_test)

	# x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_train = torch.from_numpy(x_true_train).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_true_test).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_train = torch.from_numpy(y_true_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_true_test).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test

def data_loading_with_padding_full():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
	x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')

	# Normalized

	x_train = (x_train.reshape(60000, 1024)/255. - 0.1307)/0.3081
	x_test = (x_test.reshape(10000, 1024)/255. - 0.1307)/0.3081

	# x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_test).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_test).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test


def data_loading_one_target_vs_others(target_num):
	# This function is to output data set as 
	# one-half os the target_num which is about 6000 data point
	# plus the same number randomly selected from the others
	# and the test set is divided following the same method
	# 1000 target_num and 1000 others

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
	x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')

	# Normalized

	x_train = (x_train.reshape(60000, 1024)/255. - 0.1307)/0.3081
	x_test = (x_test.reshape(10000, 1024)/255. - 0.1307)/0.3081

	# Select out the train target
	# Select out the train non-target
	# Split the data

	x_true_train = []
	y_true_train = []
	x_false_train = []
	y_false_train = []

	for idx in range(len(y_train)):
		if y_train[idx] == target_num:
			x_true_train.append(x_train[idx])
			y_true_train.append(y_train[idx])
		else:
			x_false_train.append(x_train[idx])
			y_false_train.append(y_train[idx])

	x_true_train = np.array(x_true_train)
	y_true_train = np.array(y_true_train)
	x_false_train = np.array(x_false_train)
	y_false_train = np.array(y_false_train)

	permutation_false_train = np.random.permutation(len(y_false_train))

	x_false_train = x_false_train[permutation_false_train[:len(y_true_train)]]
	y_false_train = y_false_train[permutation_false_train[:len(y_true_train)]]


	# Split the data in the test set
	# Select out the test target
	# Select out the train non-target
	x_true_test = []
	y_true_test = []
	x_false_test = []
	y_false_test = []

	for idx in range(len(y_test)):
		if y_test[idx] == target_num:
			x_true_test.append(x_test[idx])
			y_true_test.append(y_test[idx])
		else:
			x_false_test.append(x_test[idx])
			y_false_test.append(y_test[idx])

	x_true_test = np.array(x_true_test)
	y_true_test = np.array(y_true_test)
	x_false_test = np.array(x_false_test)
	y_false_test = np.array(y_false_test)

	permutation_false_test = np.random.permutation(len(y_false_test))

	x_false_test = x_false_test[permutation_false_test[:len(y_true_test)]]
	y_false_test = y_false_test[permutation_false_test[:len(y_true_test)]]


	# Combine the data

	permutation_balanced_train = np.random.permutation(len(x_true_train) * 2)
	x_train_balanced = np.concatenate((x_true_train, x_false_train), axis = 0)
	y_train_balanced = np.concatenate((y_true_train, y_false_train), axis = 0)

	x_train_balanced = x_train_balanced[permutation_balanced_train]
	y_train_balanced = y_train_balanced[permutation_balanced_train]

	permutation_balanced_test = np.random.permutation(len(x_true_test) * 2)
	x_test_balanced = np.concatenate((x_true_test, x_false_test), axis = 0)
	y_test_balanced = np.concatenate((y_true_test, y_false_test), axis = 0)

	x_test_balanced = x_test_balanced[permutation_balanced_test]
	y_test_balanced = y_test_balanced[permutation_balanced_test]



	# Transform the y label to 0 or 1
	y_train_balanced = (y_train_balanced == target_num)
	y_test_balanced = (y_test_balanced == target_num)
	
	# x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_train = torch.from_numpy(x_train_balanced).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_test_balanced).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_train = torch.from_numpy(y_train_balanced).type(torch.LongTensor)
	y_test = torch.from_numpy(y_test_balanced).type(torch.LongTensor)

	

	return x_train, y_train, x_test, y_test






def data_loading_two_target(target_1, target_2, padding = False, size_for_each_class = 0):
	# Define the first data be labeled 0
	# Define the second data be labeled 1
	x_train_first = None
	y_train_first = None
	x_test_first = None
	y_test_first = None
	x_train_second = None
	y_train_second = None
	x_test_second = None
	y_test_second = None

	if padding == True:
		x_train_first, y_train_first, x_test_first, y_test_first = data_loading_with_padding_target(target_1)
		x_train_second, y_train_second, x_test_second, y_test_second = data_loading_with_padding_target(target_2)

	else:
		x_train_first, y_train_first, x_test_first, y_test_first = data_loading_with_target(target_1)
		x_train_second, y_train_second, x_test_second, y_test_second = data_loading_with_target(target_2)

	if size_for_each_class:
		x_train_first = x_train_first[:size_for_each_class]
		y_train_first = y_train_first[:size_for_each_class]
		x_train_second = x_train_second[:size_for_each_class]
		y_train_second = y_train_second[:size_for_each_class]

	x_train_combined = np.concatenate((x_train_first, x_train_second), axis = 0)
	y_train_combined = np.concatenate((y_train_first, y_train_second), axis = 0)

	x_test_combined = np.concatenate((x_test_first, x_test_second), axis = 0)
	y_test_combined = np.concatenate((y_test_first, y_test_second), axis = 0)

	length_first_train_set = len(x_train_first)
	length_second_train_set = len(x_train_second)

	length_first_test_set = len(x_test_first)
	length_second_test_set = len(x_test_second)

	permutation_train = np.random.permutation(length_first_train_set + length_second_train_set)
	permutation_test = np.random.permutation(length_first_test_set + length_second_test_set)

	x_train_combined = x_train_combined[permutation_train]
	y_train_combined = y_train_combined[permutation_train]

	x_test_combined = x_test_combined[permutation_test]
	y_test_combined = y_test_combined[permutation_test]

	# Transform the y label to 0 or 1
	y_train_combined = (y_train_combined == target_2)
	y_test_combined = (y_test_combined == target_2)

	# x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_train = torch.from_numpy(x_train_combined).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_train = torch.from_numpy(y_train_combined).type(torch.LongTensor)

	x_test = torch.from_numpy(x_test_combined).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_test_combined).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test

def padding_data(x):
	x = np.pad(x, ((0,0),(2,2),(2,2)), 'constant')
	return x

def load_binary_with_padding(target_num_1 = 0, target_num_2 = 1):

	x_train, y_train, x_test, y_test = data_loading_two_target(target_1 = target_num_1, target_2 = target_num_2, padding = True)

	np.random.seed(0)
	num_data = len(x_train)
	num_train = int(0.75 * num_data)

	index = np.random.permutation(range(num_data))

	x_for_train = x_train[index[:num_train]]
	y_for_train = y_train[index[:num_train]]

	x_for_val = x_train[index[num_train:]]
	y_for_val = y_train[index[num_train:]]

	x_for_test = x_test
	y_for_test = y_test

	return x_for_train, y_for_train, x_for_val, y_for_val, x_for_test, y_for_test

def load_full_class_with_padding():

	x_train, y_train, x_test, y_test = data_loading_with_padding_full()

	np.random.seed(0)
	num_data = len(x_train)
	num_train = int(0.75 * num_data)

	index = np.random.permutation(range(num_data))

	x_for_train = x_train[index[:num_train]]
	y_for_train = y_train[index[:num_train]]

	x_for_val = x_train[index[num_train:]]
	y_for_val = y_train[index[num_train:]]

	x_for_test = x_test
	y_for_test = y_test

	return x_for_train, y_for_train, x_for_val, y_for_val, x_for_test, y_for_test

def main():
	# x_train, y_train, x_test, y_test = data_loading_two_target(0,1,padding = True)
	# x_train, y_train, x_test, y_test = data_loading_one_target_vs_others(target_num = 1)
	# # print("x_train_0: ", x_train_0)
	# # print("x_train_0 shape: ", x_train_0.shape)

	# # print("y_train_0: ", y_train_0)
	# # print("Y_train_0 shape: ", y_train_0.shape)

	# # print("x_train_1: ", x_train_1)
	# # print("x_train_1 shape: ", x_train_1.shape)

	# # print("y_train_1: ", y_train_1)
	# # print("Y_train_1 shape: ", y_train_1.shape)

	# print("x_train: ", x_train)
	# print("x_train.shape: ", x_train.shape)

	# print("MAX: ",x_train.max())
	# print("MIN: ",x_train.min())

	# print("y_train: ", y_train)
	# print("y_train.shape: ", y_train.shape)

	# print("x_test: ", x_test)
	# print("x_test.shape: ", x_test.shape)

	# print("y_test: ", y_test)
	# print("y_test.shape: ", y_test.shape)

	# for i in range(10):
	# 	display_img(x_train[i].numpy().reshape(32,32))

	x_0, _, _, _ = data_loading_with_padding_target(target_num = 0)

	x_1, _, _, _ = data_loading_with_padding_target(target_num = 1)

	x_0 = x_0.numpy()
	x_1 = x_1.numpy()

	x_0_square_sum = np.sum(x_0 ** 2, axis = 1).reshape(len(x_0),1)
	print(x_0_square_sum.shape)
	# x_0_sqrt = np.clip(np.sqrt(x_0_square_sum), a_min = 1e-9, a_max = None)
	x_0_sqrt = np.sqrt(x_0_square_sum)
	print(x_0_sqrt.shape)
	x_0 = x_0 / x_0_sqrt
	print(x_0.shape)

	x_1_square_sum = np.sum(x_1 ** 2, axis = 1).reshape(len(x_1),1)
	print(x_1_square_sum.shape)
	# x_1_sqrt = np.clip(np.sqrt(x_1_square_sum), a_min = 1e-9, a_max = None)
	x_1_sqrt = np.sqrt(x_1_square_sum)
	print(x_1_sqrt.shape)
	x_1 = x_1 / x_1_sqrt
	print(x_1.shape)

	x_0 = x_0.mean(axis = 0)
	x_1 = x_1.mean(axis = 0)

	plot_two_classes(x_0, x_1)
	# 2019 10 28 : MNIST DATA BEHAVE CORRECTLY EVEN WITHOUT CLIP IN THE NORMALIZATION






if __name__ == '__main__':
	main()
	