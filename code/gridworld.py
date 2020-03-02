
# standard
from numpy.random import random
from collections import namedtuple
from mdptoolbox.mdp import PolicyIterationModified
import numpy as np 

# my package 
# from agent import Agent, Service
import plotter 
import utilities 
from helper_classes import Gaussian, Agent, Service, Dispatch, Empty, CustomerModel

class GridWorld():
	def __init__(self,param):
		self.param = param
		self.name = param.env_name 
		self.timestep = 0
		self.observation = []
		self.cm = CustomerModel(param)

	def init_agents(self):
		# initialize list of agents  
		self.agents = []
		for i in range(self.param.ni):
			x,y = utilities.random_position_in_world()
			self.agents.append(Agent(i,x,y,self.v0,self.q0))
			print('agent {} initialized at (x,y) = ({},{})'.format(i,x,y))

	def render(self,title=None):
		time=self.param.sim_times[self.timestep]

		fig,ax = plotter.make_fig()
		ax.set_xticks(self.param.env_x)
		ax.set_yticks(self.param.env_y)
		ax.set_xlim(self.param.env_xlim)
		ax.set_ylim(self.param.env_ylim)
		ax.set_aspect('equal')
		if title is None:
			ax.set_title('t={}/{}'.format(time,self.param.sim_times[-1]))
		else:
			ax.set_title(title)
		ax.grid(True)

		# state space
		for agent in self.agents:
			
			color = self.param.plot_agent_mode_color[agent.mode]
			plotter.plot_circle(agent.x,agent.y,self.param.plot_r_agent,fig=fig,ax=ax,color=color)
			
			if agent.i == 0:
				plotter.plot_dashed(agent.x,agent.y,self.param.r_comm,fig=fig,ax=ax,color=color)
		
			# pickup 
			if agent.mode == 1:
				plotter.plot_rectangle(agent.service.x_p, agent.service.y_p,\
					self.param.plot_r_customer,fig=fig,ax=ax,color=self.param.plot_customer_color[agent.mode])
				plotter.plot_line(agent.x,agent.y,agent.service.x_p,agent.service.y_p,fig=fig,ax=ax,color=self.param.plot_customer_color[agent.mode])

			# dropoff 
			elif agent.mode == 2:
				plotter.plot_rectangle(agent.service.x_d, agent.service.y_d,\
					self.param.plot_r_customer,fig=fig,ax=ax,color=self.param.plot_customer_color[agent.mode])
				plotter.plot_line(agent.x,agent.y,agent.service.x_d,agent.service.y_d,fig=fig,ax=ax,color=self.param.plot_customer_color[agent.mode])

			# on dispatch 
			elif agent.mode == 3 and self.param.plot_arrows_on:	
				dx = agent.dispatch.x - agent.x
				dy = agent.dispatch.y - agent.y
				plotter.plot_arrow(agent.x,agent.y,dx,dy,fig=fig,ax=ax,color=self.param.plot_customer_color[agent.mode])

		for service in self.observation:
			# print('service: ', service)
			plotter.plot_rectangle(service.x_p,service.y_p,self.param.plot_r_customer,fig=fig,ax=ax,\
				color=self.param.plot_customer_color[1],angle=45)

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
		for i_agent, agent in enumerate(self.agents):
			action = actions[i_agent]
			wait_time += self.agent_step(agent,action)
		reward = -wait_time 
		
		# extract state with numpy arrays
		# - agent states
		agents_operation = np.empty(self.param.ni)
		agents_location = np.empty((self.param.ni,2))
		agents_value_fnc_vector = np.empty((self.param.ni,self.param.env_ncell))

		for agent in self.agents:
			agents_operation[agent.i] = agent.mode
			agents_location[agent.i,:] = [agent.x,agent.y]
			agents_value_fnc_vector[agent.i,:] = utilities.q_value_to_value_fnc(self,agent.q)

		# - customer states
		customers_location = self.get_curr_customer_locations()

		# - distributions 
		gmm_distribution = self.get_curr_im_gmm()
		agents_distribution = self.get_curr_im_agents()
		agents_value_fnc_distribution = self.get_curr_im_value()

		# put desired numpy arrays into dictionary
		state = dict()
		for state_key in self.param.state_keys:
			state[state_key] = eval(state_key)

		# increment timestep
		self.timestep += 1
		return reward, state

	def eta_cell(self,i,j,t):
		# expected time of arrival between two states (i,j), at given time (t)
		x_i,y_i = utilities.cell_index_to_cell_coordinate(i)
		x_j,y_j = utilities.cell_index_to_cell_coordinate(j)
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

		tf_train = int(self.param.n_training_data/self.param.n_customers_per_timestep)
		tf_sim = max(self.param.sim_dt,self.param.sim_tf)

		# 'move' gaussians around for full simulation time 
		self.cm.run_cm_model()

		# training dataset part 
		dataset = []
		customer_time_array_train = np.arange(-tf_train,0,self.param.sim_dt)
		for time in customer_time_array_train:
			for customer in range(self.param.n_customers_per_timestep):
				time_of_request = time + np.random.random() * self.param.sim_dt
				x_p,y_p = self.cm.sample_cm(0)
				x_d,y_d = utilities.random_position_in_world()
				time_to_complete = self.eta(x_p,y_p,x_d,y_d)
				dataset.append(np.array([time_of_request,time_to_complete,x_p,y_p,x_d,y_d]))

		# testing dataset part 
		customer_time_array_sim = np.arange(0,tf_sim,self.param.sim_dt)
		for timestep,time in enumerate(customer_time_array_sim):
			for customer in range(self.param.n_customers_per_timestep):
				time_of_request = time + np.random.random() * self.param.sim_dt
				x_p,y_p = self.cm.sample_cm(timestep)
				x_d,y_d = utilities.random_position_in_world()
				time_to_complete = self.eta(x_p,y_p,x_d,y_d)
				dataset.append(np.array([time_of_request,time_to_complete,x_p,y_p,x_d,y_d]))

		# solve mdp
		dataset = np.array(dataset)
		self.dataset = dataset

		train_dataset = dataset[dataset[:,0]<0,:]
		curr_time = 0 
		v,q = utilities.solve_MDP(self,train_dataset,curr_time)
		
		self.v0 = v
		self.q0 = q 

	def agent_step(self,agent,action):

		# agent can be assigned a service or a dispatch 
		# agent updates its state according to its operation (or mode)
		# mode: 
		# 	1: pickup mode. moves agent along pickup vector 
		# 	2: dropoff mode. moves agent along dropoff vector
		# 	3: dispatch mode. moves agent along dispatch vector

		wait_time = 0 

		# assignment 
		if isinstance(action,Service):
			agent.service = action
			agent.update = True

			curr_time = self.param.sim_times[self.timestep] 
			eta = self.eta(agent.x,agent.y,agent.service.x_p,agent.service.y_p) 
			wait_time = eta + agent.service.time_before_assignment

			# initialize service
			agent.mode = 1 # pickup mode
			agent.pickup_vector = np.array([agent.service.x_p - agent.x, agent.service.y_p-agent.y])
			agent.pickup_dist = np.linalg.norm(agent.pickup_vector)
			agent.pickup_vector = agent.pickup_vector / agent.pickup_dist
			agent.pickup_speed = agent.pickup_dist/eta
			agent.pickup_finish_time = curr_time + eta

			agent.dropoff_vector = np.array([agent.service.x_d - agent.service.x_p,agent.service.y_d-agent.service.y_p])
			agent.dropoff_dist = np.linalg.norm(agent.dropoff_vector)
			agent.dropoff_vector = agent.dropoff_vector/agent.dropoff_dist
			agent.dropoff_speed = agent.dropoff_dist/agent.service.time_to_complete
			agent.dropoff_finish_time = agent.pickup_finish_time + agent.service.time_to_complete

		elif isinstance(action,Dispatch):

			agent.dispatch = action 
			agent.mode = 3

			# assign dispatch move 
			curr_time = self.param.sim_times[self.timestep] 
			eta = self.eta(agent.x,agent.y,agent.dispatch.x,agent.dispatch.y)
			agent.move_vector = np.array([agent.dispatch.x - agent.x, agent.dispatch.y-agent.y])
			agent.move_dist = np.linalg.norm(agent.move_vector)
			agent.move_vector = agent.move_vector/agent.move_dist
			agent.move_speed = agent.move_dist/eta 
			agent.move_finish_time = curr_time+eta 
							
		# update			
		next_time = self.param.sim_times[self.timestep+1]
		time = self.param.sim_times[self.timestep]
		dt = next_time - time 

		# pickup mode 
		if agent.mode == 1:
			if next_time > agent.pickup_finish_time:
				agent.x = agent.service.x_p
				agent.y = agent.service.y_p 
				agent.mode = 2
			else:
				agent.x = agent.x + agent.pickup_vector[0]*agent.pickup_speed*dt
				agent.y = agent.y + agent.pickup_vector[1]*agent.pickup_speed*dt

		# dropoff mode 
		elif agent.mode == 2:
			if next_time > agent.dropoff_finish_time:
				agent.x = agent.service.x_d
				agent.y = agent.service.y_d 
				agent.mode = 0
			else:
				agent.x = agent.x + agent.dropoff_vector[0]*agent.dropoff_speed*dt
				agent.y = agent.y + agent.dropoff_vector[1]*agent.dropoff_speed*dt

		# dispatch mode 
		elif agent.mode == 3:
			if next_time > agent.move_finish_time:
				agent.x = agent.dispatch.x
				agent.y = agent.dispatch.y
				agent.mode = 0
			else:
				dt = next_time-time
				agent.x = agent.x + agent.move_vector[0]*agent.move_speed*dt
				agent.y = agent.y + agent.move_vector[1]*agent.move_speed*dt

		# idle mode
		elif agent.mode == 0:
			agent.x = agent.x
			agent.y = agent.y


		# make sure you don't leave environment
		agent.x,agent.y = utilities.environment_barrier([agent.x,agent.y])

		return wait_time

	def get_curr_im_value(self):

		value_fnc_ims = np.zeros((self.param.ni,self.param.env_ny,self.param.env_nx))
		for agent in self.agents:
			# get value
			value_fnc_i = utilities.q_value_to_value_fnc(self,agent.q)
			# normalize
			value_fnc_i = (value_fnc_i - min(value_fnc_i))/(max(value_fnc_i)-min(value_fnc_i))
			# convert to im
			im_v = np.zeros((self.param.env_nx,self.param.env_ny))
			for i in range(self.param.env_ncell):
				i_x,i_y = utilities.cell_index_to_xy_cell_index(i)
				im_v[i_x,i_y] = value_fnc_i[i]
			# convert to im coordinates ? 
			im_v = im_v.T
			im_v = np.flipud(im_v)
			# add
			value_fnc_ims[agent.i,:,:] = im_v
		return value_fnc_ims

	def get_curr_im_gmm(self):

		# im is [nx,ny] where im[0,0] is bottom left
		im_gmm = self.cm.eval_cm(self.timestep)
		
		# im coordinates
		im_gmm = im_gmm.T
		im_gmm = np.flipud(im_gmm)

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
			idx_x,idx_y = utilities.coordinate_to_xy_cell_index(
				locs[agent.i,0],
				locs[agent.i,1])
			im_agent[idx_x][idx_y] += 1

		# normalize
		im_agent = im_agent / self.param.ni

		# im coordinates
		im_agent = im_agent.T
		im_agent = np.flipud(im_agent)

		return im_agent

	def get_curr_im_free_agents(self,timestep):

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
				idx_x,idx_y = utilities.coordinate_to_xy_cell_index(
					locs[agent.i,0],
					locs[agent.i,1])
				im_agent[idx_x][idx_y] += 1
				count_agent += 1

		# normalize
		if count_agent > 0:
			im_agent = im_agent / count_agent

		# im coordinates
		im_agent = im_agent.T
		im_agent = np.flipud(im_agent)

		return im_agent

	def get_curr_customer_locations(self):
		
		# customers_location = np.zeros((self.param.env_nx,self.param.env_ny))
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

		# if count > 0 :
		# 	customers_location /= count

		return customers_location	