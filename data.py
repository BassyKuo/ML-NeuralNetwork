import numpy as np

def load_data (data_csv, label_index=0):
	"""
	Usage:
		x, y = load_data("train.csv")

	-x	: datapoints
	-y	: labels
	"""
	data = np.genfromtxt(data_csv, delimiter=',')

	data = np.array(data)
	y = data[:,label_index].reshape((-1,1))
	x = np.c_[data[:,:label_index], data[:,label_index+1:]]
	return x, y

def shuffle_union (x, y):
	"""
	Usage:
		x_shuffle, y_shuffle = shuffle_union(x,y)
	"""
	index = np.random.permutation(len(x))
	return x[index], y[index]
