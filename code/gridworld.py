
# standard
from numpy.random import random
from collections import namedtuple
from mdptoolbox.mdp import PolicyIterationModified,ValueIteration
import numpy as np 

# my package 
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

# 'cell_index' : element of [0,...,env_ncell]
# 'cell_coordinate' : (x,y) coordinates of bottom left corner of cell 
# 'xy_cell_index' : (i_x,i_y) indices corresponding to elements of env_x, env_y
# 'coordinate' : free (x,y) coordinate, not constrained by being on gridlines

class Utility: 
	# currently does the following:
	# 	- MDP solve
	# 	- cell index stuff 

	def __init__(self,param):
		self.param = param 

		
	def cell_index_to_cell_coordinate(self,i):
		if self.param.env_name is 'gridworld':
			x = self.param.env_dx*np.remainder(i,self.param.env_nx)
			y = self.param.env_dy*np.floor_divide(i,self.param.env_nx)
		return x,y


	def xy_cell_index_to_cell_index(self,i_x,i_y):
		i = i_y*len(self.param.env_x) + i_x
		return int(i) 


	def cell_index_to_xy_cell_index(self,i):
		x,y = self.cell_index_to_cell_coordinate(i)
		i_x,i_y = self.coordinate_to_xy_cell_index(x,y)
		return i_x,i_y


	def coordinate_to_xy_cell_index(self,x,y):
		i = self.coordinate_to_cell_index(x,y)
		x,y = self.cell_index_to_cell_coordinate(i)
		i_x = x/self.param.env_dx
		i_y = y/self.param.env_dy
		return int(i_x),int(i_y)


	def coordinate_to_cell_index(self,x,y):
		if self.param.env_name is 'gridworld':
			i_x = np.where(self.param.env_x <= x)[0][-1]
			i_y = np.where(self.param.env_y <= y)[0][-1]
			i = self.xy_cell_index_to_cell_index(i_x,i_y)
		return int(i)


	def random_position_in_cell(self,i):
		if self.param.env_name is 'gridworld':
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


	def softmax(self,x):
		softmax = np.exp(self.param.beta*x)/sum(np.exp(self.param.beta*x))
		return softmax


	def value_to_probability(self,x):
		x = x/abs(sum(x))
		x = self.softmax(x)
		return x


	def get_next_state(self,env,s,a):
		P = self.P
		next_state = np.where(P[a,s,:] == 1)[0][0]
		return next_state

	def s_sp_to_a(self,env,s,sp):
		P = self.P
		next_state = np.where(P[:,s,sp] == 1)[0][0]
		return next_state

	def sa_to_q_idx(self,s,a):
		q_idx = s*self.param.env_naction + a
		return q_idx 


	def get_prev_sa_lst(self,env,s):
		prev_sa_lst = []
		P = self.P
		local_s = get_local_states(env,s)
		added_s = []
		for prev_s in local_s:
			for prev_a in range(self.param.env_naction):
				if P[prev_a,prev_s,s] == 1 and not prev_s in added_s:
					prev_sa_lst.append((prev_s,prev_a))
					added_s.append(prev_s)
		return prev_sa_lst 


	def get_local_q_values(self,env,agent):
		local_q = []
		local_s = []
		s = self.coordinate_to_cell_index(agent.x,agent.y)
		for a in range(env.param.env_naction):
			next_state = self.get_next_state(env,s,a)
			if not next_state in local_s:
				local_s.append(next_state)
				local_q.append(agent.q[self.sa_to_q_idx(s,a)])
		return local_q


	def get_local_states(self,env,s):
		local = []
		for a in range(env.param.env_naction):
			next_state = self.get_next_state(env,s,a) 
			if not next_state in local:
				local.append(next_state)
		return local 


	def reward_instance(self,env,s,a,px,py):

		# input 
		# 	-env: 
		#	-s: current state
		# 	-a: action
		# 	-px,py: x,y position of customer data
		# output
		# 	-reward: customer waiting time 
		
		sp = self.get_next_state(env,s,a)
		spx,spy = self.cell_index_to_cell_coordinate(sp)
		spx += self.param.env_dx/2
		spy += self.param.env_dy/2
		sx,sy = self.cell_index_to_cell_coordinate(s)
		sx += self.param.env_dx/2
		sy += self.param.env_dy/2

		time_s_to_sp = env.eta(sx,sy,spx,spy)
		time_sp_to_c = env.eta(spx,spy,px,py)
		reward = -1*(time_s_to_sp + time_sp_to_c)
		
		# action_cost = param.lambda_a*(not a==0)
		# cost = cwt + action_cost 

		return reward

	# mdp stuff 
	def solve_MDP(self,env,dataset,curr_time):

		self.P = self.get_MDP_P(env) # in AxSxS
		P = self.P
		R = self.get_MDP_R(env,dataset,curr_time) # in SxA
		mdp = ValueIteration(P,R,env.param.mdp_gamma,env.param.mdp_eps,env.param.mdp_max_iter)
		# mdp = ValueIterationGS(P, R, env.param.mdp_gamma, epsilon=env.param.mdp_eps, max_iter=env.param.mdp_max_iter)
		# mdp.setVerbose()
		mdp.run()
		V = np.array(mdp.V)
		Q = self.get_MDP_Q(env,R,V,env.param.mdp_gamma)
		return V,Q  


	def get_MDP_Q(self,env,R,V,gamma):

		Q = np.zeros(env.param.env_naction*env.param.env_ncell)
		for s in range(env.param.env_ncell):
			for a in range(env.param.env_naction):
				next_state = self.get_next_state(env,s,a)
				idx = s*env.param.env_naction + a 
				Q[idx] = R[s,a] + gamma*V[next_state]
		return Q 

	def q_value_to_value_fnc(self,env,q):
		v = np.zeros(self.param.env_ncell)
		for s in range(env.param.env_ncell):
			idx = s*self.param.env_naction + np.arange(0,self.param.env_naction)
			v[s] = max(q[idx])
		return v



	def get_MDP_P(self,env):
		# P in AxSxS 
		P = np.zeros((env.param.env_naction,env.param.env_ncell,env.param.env_ncell))

		for s in range(env.param.env_ncell):

			x,y = self.cell_index_to_cell_coordinate(s)

			# print('s: ',s)
			# print('x: ',x)
			# print('y: ',y)
			
			# 'empty' action  
			P[0,s,s] = 1.

			# 'right' action
			if not x == env.param.env_x[-1]:
				P[1,s,s+1] = 1.
			else:
				P[1,s,s] = 1.

			# 'top' action
			if not y == env.param.env_y[-1]:
				next_s = self.coordinate_to_cell_index(x,y+self.param.env_dy)
				P[2,s,next_s] = 1.
			else:
				P[2,s,s] = 1.

			# 'left' action
			if not x == env.param.env_x[0]:
				P[3,s,s-1] = 1.
			else:
				P[3,s,s] = 1.			

			# 'down' action
			if not y == env.param.env_y[0]:
				next_s = self.coordinate_to_cell_index(x,y-self.param.env_dy)
				P[4,s,next_s] = 1. 
			else:
				P[4,s,s] = 1.

			# print('P[:,s,:]:', P[:,s,:])
		# exit()

		return P  


	def get_MDP_R(self,env,dataset,curr_time):
		# R in SxA
		R = np.zeros((env.param.env_ncell,env.param.env_naction))
		P = self.P

		count = 0
		time_discount_sum = np.ones(R.shape)
		for data in dataset:
			
			tor = data[0]
			time_diff = curr_time - tor

			if time_diff < 0:
				break

			count += 1
			px = data[2]
			py = data[3]

			for s in range(env.param.env_ncell):
				for a in range(env.param.env_naction):
					
					time_discount = env.param.lambda_r**time_diff
					R[s,a] += self.reward_instance(env,s,a,px,py)*time_discount
					time_discount_sum[s,a] += time_discount

		if count > 0 :
			R /= count

		R = R/time_discount_sum

		return R  
