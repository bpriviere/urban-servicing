
# standard
from numpy.random import random
from collections import namedtuple
from mdptoolbox.mdp import PolicyIterationModified,ValueIteration
import numpy as np 

# my package 
from env import Env
from helper_classes import Gaussian, Agent, Service, Dispatch, Empty, CustomerModel
import plotter

class GridWorld(Env):
	def __init__(self,param):
		super().__init__(param)
		self.init_map()
		self.cm = CustomerModel(self.param,self)
		
	def init_map(self):
		# make utility maps 
		self.grid_index_to_cell_index_map = np.zeros((self.param.env_nx,self.param.env_ny),dtype=int)
		self.cell_index_to_grid_index_map = np.zeros((self.param.env_ncell,2),dtype=int)
		count = 0
		for i_y,y in enumerate(self.param.env_y):
			for i_x,x in enumerate(self.param.env_x):
				self.grid_index_to_cell_index_map[i_x,i_y] = count
				self.cell_index_to_grid_index_map[count,:] = [i_x,i_y]
				count += 1


	def make_dataset(self):
		# make dataset that will be used for training and testing
		# training time: [tf_train,0]
		# testing time:  [0,tf_sim]

		tf_train = int(self.param.n_training_data/self.param.n_customers_per_time)
		tf_sim = max(1,int(np.ceil(self.param.sim_tf)))

		# 'move' gaussians around for full simulation time 
		self.cm.run_cm_model()

		# training dataset 
		dataset = []
		customer_time_array_train = np.arange(-tf_train,0,1,dtype=int)
		for time in customer_time_array_train:
			for customer in range(self.param.n_customers_per_time):
				time_of_request = time + np.random.random()
				x_p,y_p = self.cm.sample_cm(0)
				x_p,y_p = self.environment_barrier([x_p,y_p])
				x_d,y_d = self.random_position_in_world()
				time_to_complete = self.eta(x_p,y_p,x_d,y_d)
				dataset.append(np.array([time_of_request,time_to_complete,x_p,y_p,x_d,y_d]))

		# testing dataset 
		customer_time_array_sim = np.arange(0,tf_sim,1,dtype=int)
		for timestep,time in enumerate(customer_time_array_sim):
			sim_timestep = int(timestep/self.param.sim_dt)
			for customer in range(self.param.n_customers_per_time):
				time_of_request = time + np.random.random()
				x_p,y_p = self.cm.sample_cm(sim_timestep)
				x_p,y_p = self.environment_barrier([x_p,y_p])
				x_d,y_d = self.random_position_in_world()
				time_to_complete = self.eta(x_p,y_p,x_d,y_d)
				dataset.append(np.array([time_of_request,time_to_complete,x_p,y_p,x_d,y_d]))

		# solve mdp
		dataset = np.array(dataset)

		train_dataset = dataset[dataset[:,0] < 0,:]
		test_dataset = dataset[dataset[:,0] > 0,:]

		return train_dataset, test_dataset 

	def eta(self,x_i,y_i,x_j,y_j):
		dist = np.linalg.norm([x_i-x_j,y_i-y_j])
		return dist/self.param.taxi_speed



	# Utility stuff
		# 'cell_index' : element of [0,...,env_ncell]
		# 'cell_coordinate' : (x,y) coordinates of bottom left corner of cell 
		# 'xy_cell_index' : (i_x,i_y) indices corresponding to elements of env_x, env_y
		# 'coordinate' : free (x,y) coordinate, not constrained by being on gridlines
		# 'cell_index_to_grid_index_map' : a
		# 'grid_index_to_cell_index_map' : a 

	def random_position_in_cell(self,i):
		x,y = self.cell_index_to_cell_coordinate(i)
		x = self.param.env_dx*random() + x
		y = self.param.env_dy*random() + y
		return x,y

	def random_position_in_world(self):
		x = random()*(self.param.env_xlim[1] - self.param.env_xlim[0]) + self.param.env_xlim[0]
		y = random()*(self.param.env_ylim[1] - self.param.env_ylim[0]) + self.param.env_ylim[0]
		return x,y 

	def environment_barrier(self,p):
		eps = 1e-16
		x = np.clip(p[0],self.param.env_xlim[0]+eps,self.param.env_xlim[1]-eps)
		y = np.clip(p[1],self.param.env_ylim[0]+eps,self.param.env_ylim[1]-eps)
		return x,y
