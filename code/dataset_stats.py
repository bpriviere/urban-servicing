


from param import Param
from citymap import CityMap
import datahandler
import plotter

import numpy as np 
import matplotlib.pyplot as plt

def ave_length(dataset):
	dist = np.linalg.norm([dataset[:,4] - dataset[:,2], dataset[:,5] - dataset[:,3]],axis=0)
	return np.mean(dist[dist > 0])

def ave_duration(dataset):
	return np.mean(dataset[:,1])

def ave_customer_per_time(dataset):
	total_customers = dataset.shape[0]
	total_time = dataset[-1,0]-dataset[0,0]
	return total_customers/total_time

def ave_taxi_speed(dataset):
	ave_dist = ave_length(dataset)
	ave_time = ave_duration(dataset)
	return ave_dist/ave_time

def main():

	param = Param()
	env = CityMap(param)

	if param.make_dataset_on:
		print('   making dataset...')
		train_dataset, test_dataset = datahandler.make_dataset(env)
		datahandler.write_dataset(env, train_dataset, test_dataset)
	
	datahandler.load_dataset(env)

	train_dataset = env.train_dataset
	test_dataset = env.test_dataset

	train_w = env.get_customer_demand(train_dataset,train_dataset[-1,0])
	test_w = env.get_customer_demand(test_dataset,test_dataset[-1,0])

	print(train_dataset[0,:])
	print(np.linalg.norm([train_dataset[0,4] - train_dataset[0,2], train_dataset[0,5] - train_dataset[0,3]]))
	print(test_dataset[0,:])
	print(np.linalg.norm([test_dataset[0,4] - test_dataset[0,2], test_dataset[0,5] - test_dataset[0,3]]))
	
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

	print('train ave taxi speed: ', ave_taxi_speed(train_dataset))
	print('test ave taxi speed: ', ave_taxi_speed(test_dataset))

	print('train_dataset[0:1,:]:', train_dataset[0:1,:])
	print('test_dataset[0:1,:]:', test_dataset[0:1,:])

	fig,ax = plt.subplots()
	ax.plot(train_dataset[:,0])
	ax.set_ylabel('pickup time')
	ax.set_xlabel('customer request #')

	fig,ax = plt.subplots()
	ax.plot(test_dataset[:,0])
	ax.set_ylabel('pickup time')
	ax.set_xlabel('customer request #')	

	fig,ax = plt.subplots()
	ax.plot(train_dataset[:,1])
	ax.set_ylabel('train duration')
	ax.set_xlabel('customer request #')

	fig,ax = plt.subplots()
	ax.plot(test_dataset[:,1])
	ax.set_ylabel('test duration')
	ax.set_xlabel('customer request #')		

	fig,ax = plt.subplots()
	ax.plot(np.linalg.norm([train_dataset[:,4] - train_dataset[:,2], train_dataset[:,5] - train_dataset[:,3]],axis=0))
	ax.set_ylabel('train length')
	ax.set_xlabel('customer request #')

	fig,ax = plt.subplots()
	ax.plot(np.linalg.norm([test_dataset[:,4] - test_dataset[:,2], test_dataset[:,5] - test_dataset[:,3]],axis=0))
	ax.set_ylabel('test length')
	ax.set_xlabel('customer request #')			

	print('saving and opening figs...')
	plotter.save_figs('dataset_plots.pdf')
	plotter.open_figs('dataset_plots.pdf')


if __name__ == '__main__':
	main()