


from param import Param
from citymap import CityMap
import datahandler

import numpy as np 


def ave_length(dataset):
	dist = np.linalg.norm([dataset[:,4] - dataset[:,2], dataset[:,5] - dataset[:,3]])
	return np.mean(dist)

def ave_duration(dataset):
	return np.mean(dataset[:,1])

def ave_customer_per_time(dataset):
	total_customers = dataset.shape[0]
	total_time = dataset[-1,0]-dataset[0,0]
	return total_customers/total_time


def main():

	param = Param()
	env = CityMap(param)
	datahandler.load_dataset(env)

	train_dataset = env.train_dataset
	test_dataset = env.test_dataset
	
	# dataset = [\
		# starttime [s],
		# duration [s], 
		# x_p (longitude) [degrees], 
		# y_p (latitute) [degrees], 
		# x_d (longitude) [degrees],
		# x_d (latitute) [degrees],
		#]

	print('train_dataset.shape',train_dataset.shape)
	print('test_dataset.shape',test_dataset.shape)

	print('train ave_length: ', ave_length(train_dataset))
	print('test ave_length: ', ave_length(test_dataset))

	print('train ave_duration: ', ave_duration(train_dataset))
	print('test ave_duration: ', ave_duration(test_dataset))

	print('train ave_customer_per_time: ', ave_customer_per_time(train_dataset))
	print('test ave_customer_per_time: ', ave_customer_per_time(test_dataset))

if __name__ == '__main__':
	main()