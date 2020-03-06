
# standard
from numpy.random import random
from collections import namedtuple
from mdptoolbox.mdp import PolicyIterationModified
import numpy as np 

# my package 
from utilities import Utility
from helper_classes import Gaussian, Agent, Service, Dispatch, Empty, CustomerModel

import plotter

class GridWorld():
	def __init__(self,param):
		self.param = param
		self.name = param.env_name 
		self.timestep = 0
		self.observation = []

		self.utilities = Utility(self.param)
		self.cm = CustomerModel(self.param,self.utilities)

	def init_agents(self):
		# initialize list of agents  
		self.agents = []
		p0 = self.param.initial_covariance*np.ones((self.q0.shape))
		for i in range(self.param.ni):
			x,y = self.utilities.random_position_in_world()
			self.agents.append(Agent(i,x,y,self.v0,self.q0,p0))
			print('agent {} initialized at (x,y) = ({},{})'.format(i,x,y))

	# def render(self,title=None):
		
	# 	curr_time=self.param.sim_times[self.timestep]

	# 	fig,ax = plotter.make_fig()
	# 	ax.set_xticks(self.param.env_x)
	# 	ax.set_yticks(self.param.env_y)
	# 	ax.set_xlim(self.param.env_xlim)
	# 	ax.set_ylim(self.param.env_ylim)
	# 	ax.set_aspect('equal')
	# 	if title is None:
	# 		ax.set_title('t={}/{}'.format(curr_time,self.param.sim_times[-1]))
	# 	else:
	# 		ax.set_title(title)
	# 	ax.grid(True)

	# 	# state space
	# 	for agent in self.agents:
			
	# 		color = self.param.plot_agent_mode_color[agent.mode]
	# 		plotter.plot_circle(agent.x,agent.y,self.param.plot_r_agent,fig=fig,ax=ax,color=color)
			
	# 		if agent.i == 0:
	# 			plotter.plot_dashed(agent.x,agent.y,self.param.r_comm,fig=fig,ax=ax,color=color)
		
	# 		if True:
	# 			# dispatch 
	# 			if agent.mode == 0 and self.param.plot_arrows_on and self.timestep > 0:
	# 				if hasattr(agent,'dispatch'):
	# 					dx = agent.dispatch.x - agent.x
	# 					dy = agent.dispatch.y - agent.y
	# 					plotter.plot_arrow(agent.x,agent.y,dx,dy,fig=fig,ax=ax,color=color)

	# 			# servicing 
	# 			elif False: #agent.mode == 1:
	# 				if curr_time < agent.pickup_finish_time:
	# 					square_pickup = plotter.plot_rectangle(agent.service.x_p, agent.service.y_p,\
	# 						self.param.plot_r_customer,fig=fig,ax=ax,color=self.param.plot_customer_color)
	# 					line_to_pickup = plotter.plot_line(agent.x,agent.y,agent.service.x_p,agent.service.y_p,\
	# 						fig=fig,ax=ax,color=self.param.plot_customer_color)
	# 				elif curr_time < agent.dropoff_finish_time:
	# 					square_dropoff = plotter.plot_rectangle(agent.service.x_d, agent.service.y_d,\
	# 						self.param.plot_r_customer,fig=fig,ax=ax,color=self.param.plot_customer_color)
	# 					line_to_dropoff = plotter.plot_line(agent.x,agent.y,agent.service.x_d,agent.service.y_d,\
	# 						fig=fig,ax=ax,color=self.param.plot_customer_color)

	def reset(self):
		self.timestep = 0
		self.observation = []
		self.init_agents()


	def observe(self):
		t0 = self.param.sim_times[self.timestep]
		t1 = self.param.sim_times[self.timestep+1]
		idxs = np.multiply(self.dataset[:,0] >= t0, self.dataset[:,0] < t1, dtype=bool)
		customer_requests = self.dataset[idxs,:]
		for i in range(customer_requests.shape[0]):
			self.observation.append(Service(customer_requests[i,:]))
		return self.observation

	def step(self,actions):

		# update environment
		time = self.param.sim_times[self.timestep]
		wait_time = 0 
		# for i_agent, agent in enumerate(self.agents):
		# 	action = actions[i_agent]
		# 	wait_time += self.agent_step(agent,action)
		
		for agent,action in actions:
			wait_time += self.agent_step(agent,action)

		# penalize not serviced customers
		for service in self.observation:
			wait_time += service.time_before_assignment

		reward = -wait_time 
		
		# extract state with numpy arrays
		# - agent states
		agents_operation = np.empty(self.param.ni)
		agents_location = np.empty((self.param.ni,2))
		agents_value_fnc_vector = np.empty((self.param.ni,self.param.env_ncell))
		for agent in self.agents:
			agents_operation[agent.i] = agent.mode
			agents_location[agent.i,:] = [agent.x,agent.y]
			agents_value_fnc_vector[agent.i,:] = self.utilities.q_value_to_value_fnc(self,agent.q)

		# - agent actions
		agents_int_action = self.get_agents_int_action(actions)
		agents_vec_action = self.get_agents_vec_action(actions)

		# - customer states
		customers_location = self.get_curr_customer_locations()

		# - distributions 
		gmm_distribution = self.get_curr_im_gmm()
		agents_distribution = self.get_curr_im_agents()
		free_agents_distribution = self.get_curr_im_free_agents()
		agents_value_fnc_distribution = self.get_curr_im_value()
		agents_ave_vec_action_distribution = self.get_curr_ave_vec_action(agents_location, agents_vec_action)

		# put desired numpy arrays into dictionary
		state = dict()
		for state_key in self.param.state_keys:
			state[state_key] = eval(state_key)

		# increment timestep
		self.timestep += 1
		return reward, state

	def agent_step(self,agent,action):

		# agent can be assigned a service or a dispatch 
		# agent updates its state according to its operation (or mode)
		# mode: 
		# 	0: free mode. moves agent along move vector
		# 	1: service mode. moves agent along pickup/dropoff vector

		wait_time = 0 

		# assignment 
		if isinstance(action,Service):
			agent.mode = 1 # service mode
			agent.service = action
			agent.update = True

			curr_time = self.param.sim_times[self.timestep] 
			time_to_customer = self.eta(agent.x,agent.y,agent.service.x_p,agent.service.y_p) 
			wait_time = time_to_customer

			# initialize service
			agent.pickup_vector = np.array([agent.service.x_p - agent.x, agent.service.y_p-agent.y])
			agent.pickup_dist = np.linalg.norm(agent.pickup_vector)
			agent.pickup_vector = agent.pickup_vector / agent.pickup_dist
			agent.pickup_speed = agent.pickup_dist/time_to_customer
			agent.pickup_finish_time = curr_time + time_to_customer

			agent.dropoff_vector = np.array([agent.service.x_d - agent.service.x_p,agent.service.y_d-agent.service.y_p])
			agent.dropoff_dist = np.linalg.norm(agent.dropoff_vector)
			agent.dropoff_vector = agent.dropoff_vector/agent.dropoff_dist
			agent.dropoff_speed = agent.dropoff_dist/agent.service.time_to_complete
			agent.dropoff_finish_time = agent.pickup_finish_time + agent.service.time_to_complete

		elif isinstance(action,Dispatch):

			agent.mode = 0
			agent.dispatch = action
			agent.update = False

			# assign dispatch move 
			curr_time = self.param.sim_times[self.timestep] 
			time_to_dispatch = self.eta(agent.x,agent.y,agent.dispatch.x,agent.dispatch.y)
			agent.move_vector = np.array([agent.dispatch.x - agent.x, agent.dispatch.y-agent.y])
			agent.move_dist = np.linalg.norm(agent.move_vector)
			agent.move_vector = agent.move_vector/agent.move_dist
			agent.move_speed = agent.move_dist/time_to_dispatch
			agent.move_finish_time = curr_time + time_to_dispatch


		# update state
		curr_time = self.param.sim_times[self.timestep]
		next_time = self.param.sim_times[self.timestep+1]
		time = self.param.sim_times[self.timestep]
		dt = next_time - time 

		# dispatch mode 
		if agent.mode == 0:
			if next_time > agent.move_finish_time:
				agent.x = agent.dispatch.x
				agent.y = agent.dispatch.y
			else:
				dt = next_time-time
				agent.x = agent.x + agent.move_vector[0]*agent.move_speed*dt
				agent.y = agent.y + agent.move_vector[1]*agent.move_speed*dt

		# service mode 
		elif agent.mode == 1:
			
			# pickup mode 
			if curr_time < agent.pickup_finish_time:
				
				# finish
				if next_time > agent.pickup_finish_time:
					agent.x = agent.service.x_p
					agent.y = agent.service.y_p 
				
				# continue
				else:
					agent.x += agent.pickup_vector[0]*agent.pickup_speed*dt
					agent.y += agent.pickup_vector[1]*agent.pickup_speed*dt

			# dropoff mode 
			elif curr_time < agent.dropoff_finish_time:

				# finish
				if next_time > agent.dropoff_finish_time:
					agent.x = agent.service.x_d
					agent.y = agent.service.y_d 
					agent.mode = 0

				# continue
				else:
					agent.x += agent.dropoff_vector[0]*agent.dropoff_speed*dt
					agent.y += agent.dropoff_vector[1]*agent.dropoff_speed*dt

			# exit
			elif curr_time > agent.dropoff_finish_time:
				agent.x = agent.service.x_d
				agent.y = agent.service.y_d 
				agent.mode = 0

		# make sure you don't leave environment
		agent.x,agent.y = self.utilities.environment_barrier([agent.x,agent.y])

		return wait_time		

	def eta_cell(self,i,j,t):
		# expected time of arrival between two states (i,j), at given time (t)
		x_i,y_i = self.utilities.cell_index_to_cell_coordinate(i)
		x_j,y_j = self.utilities.cell_index_to_cell_coordinate(j)
		x_i += self.param.env_dx/2
		x_j += self.param.env_dx/2
		y_i += self.param.env_dy/2
		y_j += self.param.env_dy/2
		return self.eta(x_i,y_i,x_j,y_j,t)

	def eta(self,x_i,y_i,x_j,y_j):
		dist = np.linalg.norm([x_i-x_j,y_i-y_j])
		return dist/self.param.taxi_speed

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
				x_d,y_d = self.utilities.random_position_in_world()
				time_to_complete = self.eta(x_p,y_p,x_d,y_d)
				dataset.append(np.array([time_of_request,time_to_complete,x_p,y_p,x_d,y_d]))

		# testing dataset part 
		customer_time_array_sim = np.arange(0,tf_sim,1,dtype=int)
		for timestep,time in enumerate(customer_time_array_sim):
			sim_timestep = int(timestep/self.param.sim_dt)
			for customer in range(self.param.n_customers_per_time):
				time_of_request = time + np.random.random()
				x_p,y_p = self.cm.sample_cm(sim_timestep)
				x_d,y_d = self.utilities.random_position_in_world()
				time_to_complete = self.eta(x_p,y_p,x_d,y_d)
				dataset.append(np.array([time_of_request,time_to_complete,x_p,y_p,x_d,y_d]))

		# solve mdp
		dataset = np.array(dataset)
		self.dataset = dataset

		train_dataset = dataset[dataset[:,0]<0,:]
		curr_time = 0 
		v,q = self.utilities.solve_MDP(self,train_dataset,curr_time)
		
		self.v0 = v
		self.q0 = q 

	def get_curr_im_value(self):

		value_fnc_ims = np.zeros((self.param.ni,self.param.env_nx,self.param.env_ny))
		for agent in self.agents:
			# get value
			value_fnc_i = self.utilities.q_value_to_value_fnc(self,agent.q)
			# normalize
			value_fnc_i = (value_fnc_i - min(value_fnc_i))/(max(value_fnc_i)-min(value_fnc_i))
			# convert to im
			im_v = np.zeros((self.param.env_nx,self.param.env_ny))
			for i in range(self.param.env_ncell):
				i_x,i_y = self.utilities.cell_index_to_xy_cell_index(i)
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
			idx_x,idx_y = self.utilities.coordinate_to_xy_cell_index(
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
				idx_x,idx_y = self.utilities.coordinate_to_xy_cell_index(
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
		idxs = np.multiply(self.dataset[:,0] >= t0, self.dataset[:,0] < t1, dtype=bool)
		count = 0 
		for data in self.dataset[idxs,:]:
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
				s = self.utilities.coordinate_to_cell_index(agent.x,agent.y)
				sp = self.utilities.coordinate_to_cell_index(action.x,action.y)
				int_a = self.utilities.s_sp_to_a(self,s,sp)
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
				sx,sy = self.utilities.cell_index_to_cell_coordinate(
					self.utilities.coordinate_to_cell_index(agent.x,agent.y))
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
			idx_x,idx_y = self.utilities.coordinate_to_xy_cell_index(locs[i][0],locs[i][1])
			im_a[idx_x,idx_y,:] += vec_action[i][:]
			count[idx_x,idx_y] += 1

		idx = np.nonzero(count)
		# im_a[idx] = (im_a[idx].T/count[idx]).T
		im_a[idx] = im_a[idx]/count[idx]
		return im_a		