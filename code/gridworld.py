
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

	# def init_cm(self):
	# 	# initialize customer model GMM
	# 	nt = len(self.param.sim_times) 
	# 	cgm_lst = []
	# 	for i in range(self.param.cm_ng):
	# 		if False:
	# 			x0,y0 = utilities.random_position_in_world()
	# 		else:
	# 			x0,y0 = [self.param.env_x[1]/2, self.param.env_dy/2]
	# 		cgm_lst.append(
	# 			Gaussian(i,x0,y0,self.param.cm_sigma,self.param.cm_speed, nt))
	# 		print('cgm {} initialized at (x,y) = ({},{})'.format(i,x0,y0))
	# 	self.cgm_lst = cgm_lst


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

		time = self.param.sim_times[self.timestep]
		wait_time = 0 
		for i_agent, agent in enumerate(self.agents):
			action = actions[i_agent]
			wait_time += self.agent_step(agent,action)
		reward = -wait_time 
		
		AgentState = namedtuple('AgentState',['agent_operation','agent_locations','agent_q_values'])
		agent_operation = np.empty(self.param.ni)
		agent_locations = np.empty((self.param.ni,2))
		agent_q_values = np.empty((self.param.ni,self.param.nq))
		for agent in self.agents:
			agent_operation[agent.i] = agent.mode
			agent_locations[agent.i,:] = [agent.x,agent.y]
			agent_q_values[agent.i,:] = agent.q

		agent_state = AgentState._make((agent_operation,agent_locations,agent_q_values))

		self.timestep += 1

		return reward, agent_state

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

	# def sample_cm(self,timestep):
	# 	# sample multimodal gaussian model

	# 	# weight vector 
	# 	w = np.ones((self.param.cm_ng))/self.param.cm_ng
	# 	# sample w 
	# 	i = np.random.choice(self.param.cm_ng,p=w)
	# 	# sample ith gaussian model of cgm_lst
	# 	x,y = self.cgm_lst[i].sample(timestep)
	# 	return x,y


	# def move_cm(self,timestep):
	# 	# move gaussians

	# 	dt = self.param.sim_dt 
	# 	for cgm in self.cgm_lst:
	# 		if False:
	# 			th = np.random.random()*2*np.pi
	# 		else:
	# 			th = 0 
	# 		unit_vec = np.array([np.cos(th),np.sin(th)])
	# 		move = cgm.v*dt*unit_vec
	# 		p = [cgm.x[timestep] + move[0],cgm.y[timestep] + move[1]]
	# 		p = utilities.environment_barrier(p)
	# 		safe_move = [p[0] - cgm.x[timestep], p[1] - cgm.y[timestep]]
	# 		cgm.move(safe_move,timestep)

	# def run_cm_model(self):
	# 	for step,t in enumerate(self.param.sim_times[:-1]):
	# 		self.move_cm(step)

	# def eval_cm(self,timestep):
	# 	# input: 
	# 	# 	- self : env
	# 	# 	- t : time (OR TIMESTEP???)
	# 	# output: 
	# 	# 	- cm : customer model probability matrix with shape: (env_nx,env_ny), where sum(sum(cm)) = 1 

	# 	# for cgm in self.cgm_lst:
	# 	# 	print('(cgm.x,cgm.y) = ({},{})'.format(cgm.x,cgm.y))

	# 	cm = np.zeros((self.param.env_nx,self.param.env_ny))
	# 	for i in range(self.param.cm_nsample_cm):
	# 		x,y = self.sample_cm(timestep)
	# 		x,y = utilities.environment_barrier([x,y])
	# 		i_x,i_y = utilities.coordinate_to_xy_cell_index(x,y)
	# 		cm[i_x,i_y] += 1

	# 	# normalize
	# 	cm = cm/sum(sum(cm))
	# 	return cm 

	def make_dataset(self):
		# make dataset that will be used for training and testing
		# training time: [tf_train,0]
		# testing time:  [0,tf_sim]

		tf_train = int(self.param.n_training_data/self.param.n_customers_per_time)
		tf_sim = max(1,int(self.param.sim_times[-1]))

		# 'move' gaussians around for full simulation time 
		self.cm.run_cm_model()

		# training dataset part 
		dataset = []
		customer_time_array_train = np.arange(-tf_train,0,1,dtype=int)
		for time in customer_time_array_train:
			for customer in range(self.param.n_customers_per_time):
				time_of_request = time + np.random.random()
				x_p,y_p = self.cm.sample_cm(0)
				x_d,y_d = utilities.random_position_in_world()
				time_to_complete = self.eta(x_p,y_p,x_d,y_d)
				dataset.append(np.array([time_of_request,time_to_complete,x_p,y_p,x_d,y_d]))

		# testing dataset part 
		customer_time_array_sim = np.arange(0,tf_sim,1,dtype=int)
		for timestep,time in enumerate(customer_time_array_sim):
			for customer in range(self.param.n_customers_per_time):
				time_of_request = time + np.random.random()
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

		wait_time = 0 
		if isinstance(action,Service):
			agent.service = action
			time = self.param.sim_times[self.timestep] 
			eta = self.eta(agent.x,agent.y,agent.service.x_p,agent.service.y_p) 
			wait_time = eta + agent.service.time_before_assignment
			agent.update = True

			# initialize service
			agent.mode = 1
			agent.pickup_vector = np.array([agent.service.x_p - agent.x, agent.service.y_p-agent.y])
			agent.pickup_dist = np.linalg.norm(agent.pickup_vector)
			agent.pickup_vector = agent.pickup_vector / agent.pickup_dist
			agent.pickup_speed = agent.pickup_dist/eta
			agent.pickup_finish = time + eta

			agent.dropoff_vector = np.array([agent.service.x_d - agent.service.x_p,agent.service.y_d-agent.service.y_p])
			agent.dropoff_dist = np.linalg.norm(agent.dropoff_vector)
			agent.dropoff_vector = agent.dropoff_vector/agent.dropoff_dist
			agent.dropoff_speed = agent.dropoff_dist/agent.service.time_to_complete
			agent.dropoff_finish = agent.pickup_finish + agent.service.time_to_complete

			# start moving
			next_time = self.param.sim_times[self.timestep+1]
			if next_time > agent.pickup_finish:
				agent.x = agent.service.x_p
				agent.y = agent.service.y_p 
				agent.mode = 2
			else:
				dt = next_time-time
				agent.x = agent.x + agent.pickup_vector[0]*agent.pickup_speed*dt
				agent.y = agent.y + agent.pickup_vector[1]*agent.pickup_speed*dt

		elif isinstance(action,Dispatch):

			agent.dispatch = action 
			agent.mode = 3

			# assign dispatch move 
			time = self.param.sim_times[self.timestep] 
			eta = self.eta(agent.x,agent.y,agent.dispatch.x,agent.dispatch.y)
			agent.move_vector = np.array([agent.dispatch.x - agent.x, agent.dispatch.y-agent.y])
			agent.move_dist = np.linalg.norm(agent.move_vector)
			agent.move_vector = agent.move_vector/agent.move_dist
			agent.move_speed = agent.move_dist/eta 
			agent.move_finish = time+eta 

			# start moving 
			next_time = self.param.sim_times[self.timestep+1]
			if next_time > agent.move_finish:
				agent.x = agent.dispatch.x
				agent.y = agent.dispatch.y
				agent.mode = 0
			else:
				dt = next_time-time
				agent.x = agent.x + agent.move_vector[0]*agent.move_speed*dt
				agent.y = agent.y + agent.move_vector[1]*agent.move_speed*dt
							
		elif isinstance(action,Empty):
			
			next_time = self.param.sim_times[self.timestep+1]
			time = self.param.sim_times[self.timestep]
			dt = next_time - time 

			# pickup mode 
			if agent.mode == 1:
				if next_time > agent.pickup_finish:
					agent.x = agent.service.x_p
					agent.y = agent.service.y_p 
					agent.mode = 2
				else:
					agent.x = agent.x + agent.pickup_vector[0]*agent.pickup_speed*dt
					agent.y = agent.y + agent.pickup_vector[1]*agent.pickup_speed*dt

			# dropoff mode 
			elif agent.mode == 2:
				if next_time > agent.dropoff_finish:
					agent.x = agent.service.x_d
					agent.y = agent.service.y_d 
					agent.mode = 0
				else:
					agent.x = agent.x + agent.dropoff_vector[0]*agent.dropoff_speed*dt
					agent.y = agent.y + agent.dropoff_vector[1]*agent.dropoff_speed*dt

			# dispatch mode 
			elif agent.mode == 3:
				if next_time > agent.move_finish:
					agent.x = agent.dispatch.x
					agent.y = agent.dispatch.y
					agent.mode = 0
				else:
					dt = next_time-time
					agent.x = agent.x + agent.move_vector[0]*agent.move_speed*dt
					agent.y = agent.y + agent.move_vector[1]*agent.move_speed*dt

		# make sure you don't leave environment
		agent.x,agent.y = utilities.environment_barrier([agent.x,agent.y])

		return wait_time


	def get_im_v(self,timestep,results):

		agent_idx = 0
		q_values = results.agent_q_values[timestep,agent_idx,:]
		
		# im is [nx,ny] where im[0,0] is bottom left
		v = utilities.q_value_to_value_fnc(self,q_values)
		# normalize here
		# v = v/min(v) 
		v = (v-min(v))/(max(v)-min(v))
		
		im_v = np.zeros((self.param.env_nx,self.param.env_ny))
		for i in range(self.param.env_ncell):
			i_x,i_y = utilities.cell_index_to_xy_cell_index(i)
			im_v[i_x,i_y] = v[i]
		
		im_v = im_v.T
		im_v = np.flipud(im_v)
		return im_v

	def get_im_gmm(self,timestep):

		# im is [nx,ny] where im[0,0] is bottom left
		im_gmm = self.cm.eval_cm(timestep)
		
		# im coordinates
		im_gmm = im_gmm.T
		im_gmm = np.flipud(im_gmm)

		return im_gmm

	def get_im_agent(self,timestep,results):

		locs = results.agent_locations[timestep,:,:]

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

	def get_im_free_agent(self,timestep,results):

		locs = results.agent_locations[timestep,:,:]
		modes = results.agent_operation[timestep,:]

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