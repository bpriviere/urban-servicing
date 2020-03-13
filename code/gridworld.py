
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

		# training dataset part 
		dataset = []
		customer_time_array_train = np.arange(-tf_train,0,1,dtype=int)
		for time in customer_time_array_train:
			for customer in range(self.param.n_customers_per_time):
				time_of_request = time + np.random.random()
				x_p,y_p = self.cm.sample_cm(0)
				x_d,y_d = self.random_position_in_world()
				time_to_complete = self.eta(x_p,y_p,x_d,y_d)
				dataset.append(np.array([time_of_request,time_to_complete,x_p,y_p,x_d,y_d]))

		# testing dataset part 
		customer_time_array_sim = np.arange(0,tf_sim,1,dtype=int)
		for timestep,time in enumerate(customer_time_array_sim):
			sim_timestep = int(timestep/self.param.sim_dt)
			for customer in range(self.param.n_customers_per_time):
				time_of_request = time + np.random.random()
				x_p,y_p = self.cm.sample_cm(sim_timestep)
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

	# ----- plotting -----
	def get_curr_im_value(self):

		value_fnc_ims = np.zeros((self.param.ni,self.param.env_nx,self.param.env_ny))
		for agent in self.agents:
			# get value
			value_fnc_i = self.q_value_to_value_fnc(agent.q)
			# normalize
			value_fnc_i = (value_fnc_i - min(value_fnc_i))/(max(value_fnc_i)-min(value_fnc_i))
			# convert to im
			im_v = np.zeros((self.param.env_nx,self.param.env_ny))
			for i in range(self.param.env_ncell):
				i_x,i_y = self.cell_index_to_grid_index_map[i]
				im_v[i_x,i_y] = value_fnc_i[i]
			# add
			value_fnc_ims[agent.i,:,:] = im_v
		return value_fnc_ims

	def get_curr_im_gmm(self):
		# im is [nx,ny] where im[0,0] is bottom left
		im_gmm = self.cm.eval_cm(self.timestep)
		return im_gmm

	def get_curr_im_agents(self):

		locs = np.zeros((self.param.ni,2))
		for agent in self.agents:
			locs[agent.i,0] = agent.x
			locs[agent.i,1] = agent.y

		# im is [nx,ny] where im[0,0] is bottom left
		im_agent = np.zeros((self.param.env_nx,self.param.env_ny))
		for agent in self.agents:
			i = agent.i
			idx_x,idx_y = self.coordinate_to_grid_index(
				locs[agent.i,0],
				locs[agent.i,1])
			im_agent[idx_x][idx_y] += 1

		# normalize
		im_agent = im_agent / self.param.ni

		return im_agent

	def get_curr_im_free_agents(self):

		locs = np.zeros((self.param.ni,2))
		modes = np.zeros((self.param.ni))
		for agent in self.agents:
			locs[agent.i,0] = agent.x
			locs[agent.i,1] = agent.y
			modes[agent.i] = agent.mode

		# im is [nx,ny] where im[0,0] is bottom left
		im_agent = np.zeros((self.param.env_nx,self.param.env_ny))
		count_agent = 0
		for agent in self.agents:
			i = agent.i
			if modes[i] in [0,3]:
				idx_x,idx_y = self.coordinate_to_grid_index(
					locs[agent.i,0],
					locs[agent.i,1])
				im_agent[idx_x][idx_y] += 1
				count_agent += 1

		# normalize
		if count_agent > 0:
			im_agent = im_agent / count_agent

		return im_agent

	def get_curr_customer_locations(self):		
		customers_location = []
		t0 = self.param.sim_times[self.timestep]
		t1 = self.param.sim_times[self.timestep+1]
		idxs = np.multiply(self.test_dataset[:,0] >= t0, self.test_dataset[:,0] < t1, dtype=bool)
		count = 0 
		for data in self.test_dataset[idxs,:]:
			count += 1

			tor = data[0]
			px = data[2]
			py = data[3]

			customers_location.append([px,py])

		customers_location = np.asarray(customers_location)
		return customers_location	

	def get_agents_int_action(self,actions):
		# pass in list of objects
		# pass out list of integers 
		int_actions_lst = []
		# for i,agent in enumerate(self.agents):
		# 	action = actions[i]
		for agent,action in actions:
			if isinstance(action,Dispatch): 
				s = self.coordinate_to_cell_index(agent.x,agent.y)
				sp = self.coordinate_to_cell_index(action.x,action.y)
				int_a = self.s_sp_to_a(s,sp)
				int_actions_lst.append(int_a)
			elif isinstance(action,Service) or isinstance(action,Empty):
				int_actions_lst.append(-1)
			else:
				exit('get_agents_int_action type error')
		return np.asarray(int_actions_lst)

	def get_agents_vec_action(self,actions):
		# pass in list of action objects
		# pass out np array of move vectors

		agents_vec_action = np.zeros((self.param.ni,2))
		# for i,agent in enumerate(agents):
		# action = action
		for agent,action in actions:
			if isinstance(action,Dispatch):
				# agents_vec_action[i,0] = action.x - agent.x
				# agents_vec_action[i,1] = action.y - agent.y
				sx,sy = self.cell_index_to_cell_coordinate(
					self.coordinate_to_cell_index(agent.x,agent.y))
				agents_vec_action[agent.i,0] = action.x - (sx + self.param.env_dx/2)
				agents_vec_action[agent.i,1] = action.y - (sy + self.param.env_dy/2)

			elif isinstance(action,Service) or isinstance(action,Empty):
				agents_vec_action[agent.i,0] = 0
				agents_vec_action[agent.i,1] = 0
			else:
				print(action)
				exit('get_agents_vec_action type error')
		return agents_vec_action

	def get_curr_ave_vec_action(self,locs,vec_action):
		# locs is (ni,2)
		# vec_actions is (ni,2)
		ni = locs.shape[0]
		im_a = np.zeros((self.param.env_nx,self.param.env_ny,2))
		count = np.zeros((self.param.env_nx,self.param.env_ny,1))
		for i in range(ni):
			idx_x,idx_y = self.coordinate_to_grid_index(locs[i][0],locs[i][1])
			im_a[idx_x,idx_y,:] += vec_action[i][:]
			count[idx_x,idx_y] += 1

		idx = np.nonzero(count)
		# im_a[idx] = (im_a[idx].T/count[idx]).T
		im_a[idx] = im_a[idx]/count[idx]
		return im_a

	# 'cell_index' : element of [0,...,env_ncell]
	# 'cell_coordinate' : (x,y) coordinates of bottom left corner of cell 
	# 'xy_cell_index' : (i_x,i_y) indices corresponding to elements of env_x, env_y
	# 'coordinate' : free (x,y) coordinate, not constrained by being on gridlines
	# 'cell_index_to_grid_index_map' : a
	# 'grid_index_to_cell_index_map' : a 

	def cell_index_to_cell_coordinate(self,i):
		# takes in valid cell index and returns bottom left corner coordinate of cell
		# dim(self.cell_index_to_cell_coordinate_map) = [nvalidcells, 2]
		i_x,i_y = self.cell_index_to_grid_index_map[i,:]
		x,y = self.grid_index_to_coordinate(i_x,i_y)
		return x,y

	def coordinate_to_grid_index(self,x,y):
		# takes in coordinate and returns which i_x,i_y cell it is in
		i_x = np.where(self.param.env_x <= x)[0][-1] # last index where input-x is larger than grid 
		i_y = np.where(self.param.env_y <= y)[0][-1]
		return i_x,i_y

	def coordinate_to_cell_index(self,x,y):
		i_x,i_y = self.coordinate_to_grid_index(x,y)
		i = self.grid_index_to_cell_index_map[i_x,i_y] # i should always be a valid index 
		return i

	def grid_index_to_coordinate(self,i_x,i_y):
		x = self.param.env_x[i_x]
		y = self.param.env_y[i_y]
		return x,y

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

	def get_MDP_P(self):
		# P in AxSxS 
		P = np.zeros((self.param.env_naction,self.param.env_ncell,self.param.env_ncell))

		for s in range(self.param.env_ncell):

			x,y = self.cell_index_to_cell_coordinate(s)

			# print('s: ',s)
			# print('x: ',x)
			# print('y: ',y)
			
			# 'empty' action  
			P[0,s,s] = 1.

			# 'right' action
			if not x == self.param.env_x[-1]:
				P[1,s,s+1] = 1.
			else:
				P[1,s,s] = 1.

			# 'top' action
			if not y == self.param.env_y[-1]:
				next_s = self.coordinate_to_cell_index(x,y+self.param.env_dy)
				P[2,s,next_s] = 1.
			else:
				P[2,s,s] = 1.

			# 'left' action
			if not x == self.param.env_x[0]:
				P[3,s,s-1] = 1.
			else:
				P[3,s,s] = 1.			

			# 'down' action
			if not y == self.param.env_y[0]:
				next_s = self.coordinate_to_cell_index(x,y-self.param.env_dy)
				P[4,s,next_s] = 1. 
			else:
				P[4,s,s] = 1.

			# print('P[:,s,:]:', P[:,s,:])
		# exit()

		return P  