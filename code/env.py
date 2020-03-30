
# standard
from numpy.random import random
from collections import namedtuple
from mdptoolbox.mdp import PolicyIterationModified,ValueIteration
import numpy as np 

# my package 
# from utilities import Utility
from helper_classes import Gaussian, Agent, Service, Dispatch, Empty, CustomerModel

import plotter

class Env():
	# superclass of 'citymap' and 'gridworld'
	def __init__(self,param):
		self.param = param
		self.name = param.env_name 
		self.timestep = 0
		self.observation = []

	def init_agents(self,s0):
		
		# initialize list of agents  
		self.agents = []
		p0 = self.param.p0 
		for i in range(self.param.ni):
			self.agents.append(Agent(i,s0[0,i],s0[1,i],self.v,self.q,self.r,p0))
			print('agent {} initialized at (x,y) = ({},{})'.format(i,s0[0,i],s0[1,i]))
		return s0

	def get_s0(self):
		s0 = np.zeros((2,self.param.ni))
		for i in range(self.param.ni):
			x,y = self.random_position_in_world()
			s0[:,i] = [x,y]
		return s0

	def reset(self,s0):
		self.timestep = 0
		self.observation = []
		v,q,r = self.solve_MDP(self.train_dataset,self.param.sim_times[self.timestep])
		s0 = self.init_agents(s0)
		return s0

	def observe(self):
		t0 = self.param.sim_times[self.timestep]
		t1 = self.param.sim_times[self.timestep+1]
		idxs = np.multiply(self.test_dataset[:,0] >= t0, self.test_dataset[:,0] < t1, dtype=bool)
		customer_requests = self.test_dataset[idxs,:]
		for i in range(customer_requests.shape[0]):
			self.observation.append(Service(customer_requests[i,:]))
		return self.observation

	def step(self,actions):

		# update environment
		time = self.param.sim_times[self.timestep]
		wait_time = 0 
		
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
		agents_q_value = np.empty((self.param.ni,self.param.nq))
		for agent in self.agents:
			agents_operation[agent.i] = agent.mode # mode = 0 -> on dispatch 
			agents_location[agent.i,:] = [agent.x,agent.y]
			agents_value_fnc_vector[agent.i,:] = self.q_value_to_value_fnc(agent.q)
			agents_q_value[agent.i,:] = agent.q

		# - agent actions
		# agents_int_action = self.get_agents_int_action(actions)
		# agents_vec_action = self.get_agents_vec_action(actions)

		# - customer states
		customers_location = self.get_curr_customer_locations()

		# - distributions 
		gmm_distribution = self.get_curr_im_gmm()
		agents_distribution = self.get_curr_im_agents()
		free_agents_distribution = self.get_curr_im_free_agents()
		agents_value_fnc_distribution = self.get_curr_im_value()
		# agents_ave_vec_action_distribution = self.get_curr_ave_vec_action(agents_location, agents_vec_action)

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

			if agent.x == agent.service.x_p and agent.y == agent.service.y_p:
				agent.x += 0.0001
				agent.y += 0.0001

			time_to_customer = self.eta(agent.x,agent.y,agent.service.x_p,agent.service.y_p) 
			wait_time = time_to_customer

			# initialize service
			agent.pickup_vector = np.array([agent.service.x_p - agent.x, agent.service.y_p-agent.y])
			agent.pickup_dist = np.linalg.norm(agent.pickup_vector)
			agent.pickup_vector = agent.pickup_vector / agent.pickup_dist
			agent.pickup_speed = agent.pickup_dist/time_to_customer
			agent.pickup_finish_time = curr_time + time_to_customer

			if agent.service.x_p == agent.service.x_d and agent.service.y_p == agent.service.y_d:
				agent.service.x_p += 0.0001
				agent.service.y_p += 0.0001

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
		agent.x,agent.y = self.environment_barrier([agent.x,agent.y])

		return wait_time		

	# utility stuff 
	def softmax(self,x):
		softmax = np.exp(self.param.ta_beta*x)/sum(np.exp(self.param.ta_beta*x))
		return softmax

	def value_to_probability(self,x):
		x = x/abs(sum(x))
		x = self.softmax(x)
		return x

	# MDP
	def get_next_state(self,s,a):
		P = self.P
		next_state = np.where(P[a,s,:] == 1)[0][0]
		return next_state

	def s_sp_to_a(self,s,sp):
		# not unique 
		P = self.P
		next_state = np.where(P[:,s,sp] == 1)[0][0]
		return next_state

	def sa_to_q_idx(self,s,a):
		q_idx = s*self.param.env_naction + a
		return q_idx 

	def get_prev_sa_lst(self,s):
		prev_sa_lst = []
		P = self.P
		local_s = self.get_local_states(s)
		added_s = []
		for prev_s in local_s:
			for prev_a in range(self.param.env_naction):
				if P[prev_a,prev_s,s] == 1 and not prev_s in added_s:
					prev_sa_lst.append((prev_s,prev_a))
					added_s.append(prev_s)
		return prev_sa_lst 

	def get_local_q_values(self,agent):
		local_q = []
		local_s = []
		s = self.coordinate_to_cell_index(agent.x,agent.y)
		for a in range(self.param.env_naction):
			next_state = self.get_next_state(s,a)
			if not next_state in local_s:
				local_s.append(next_state)
				local_q.append(agent.q[self.sa_to_q_idx(s,a)])
		return local_q


	def get_local_states(self,s):
		local = []
		for a in range(self.param.env_naction):
			next_state = self.get_next_state(s,a) 
			if not next_state in local:
				local.append(next_state)
		return local 


	def reward_instance(self,s,a,px,py,time_diff):

		# input 
		# 	-env: 
		#	-s: current state
		# 	-a: action
		# 	-px,py: x,y position of customer data
		# output
		# 	-reward: customer waiting time 
		
		sp = self.get_next_state(s,a)
		spx,spy = self.cell_index_to_cell_coordinate(sp)
		spx += self.param.env_dx/2
		spy += self.param.env_dy/2
		sx,sy = self.cell_index_to_cell_coordinate(s)
		sx += self.param.env_dx/2
		sy += self.param.env_dy/2

		time_s_to_sp = self.eta(sx,sy,spx,spy)
		time_sp_to_c = self.eta(spx,spy,px,py)

		# reward = -1*(time_s_to_sp + time_sp_to_c)
		reward = 1/(time_s_to_sp + time_sp_to_c)
		
		# reward = -1 * time_sp_to_c
		# reward = 1 / time_sp_to_c

		# discount 
		time_discount = self.param.lambda_r**time_diff
		reward = reward * time_discount
		
		# action_cost = param.lambda_a*(not a==0)
		# cost = cwt + action_cost 

		return reward 

	# mdp stuff 
	def solve_MDP(self,dataset,curr_time):

		self.P = self.get_MDP_P() # in AxSxS
		# self.Pq = self.get_MDP_Pq()

		P = self.P
		R = self.get_MDP_R(dataset,curr_time) # in SxA
		mdp = ValueIteration(P,R,self.param.mdp_gamma,self.param.mdp_eps,self.param.mdp_max_iter)
		# mdp = ValueIterationGS(P, R, self.param.mdp_gamma, epsilon=self.param.mdp_eps, max_iter=self.param.mdp_max_iter)
		# mdp.setVerbose()
		mdp.run()
		V = np.array(mdp.V)
		Q = self.get_MDP_Q(R,V,self.param.mdp_gamma)

		self.v = V 
		self.q = Q 
		self.r = R.flatten()
		return self.v,self.q,self.r


	def get_MDP_Q(self,R,V,gamma):

		Q = np.zeros(self.param.env_naction*self.param.env_ncell)
		for s in range(self.param.env_ncell):
			for a in range(self.param.env_naction):
				next_state = self.get_next_state(s,a)
				idx = s*self.param.env_naction + a 
				Q[idx] = R[s,a] + gamma*V[next_state]
		return Q 

	def q_value_to_value_fnc(self,q):
		v = np.zeros(self.param.env_ncell)
		for s in range(self.param.env_ncell):
			idx = s*self.param.env_naction + np.arange(0,self.param.env_naction)
			v[s] = max(q[idx])
		return v

	def get_MDP_R(self,dataset,curr_time):
		# R in SxA
		R = np.zeros((self.param.env_ncell,self.param.env_naction))
		P = self.P

		count = 0
		time_discount_sum = np.zeros(R.shape)
		for data in dataset:
			
			tor = data[0]
			time_diff = curr_time - tor

			if time_diff < 0:
				break

			count += 1
			px = data[2]
			py = data[3]

			# if global update 
			if self.param.global_reward_on:
				for s in range(self.param.env_ncell):
					for a in range(self.param.env_naction):
						
						# time_discount = self.param.lambda_r**time_diff
						# R[s,a] += self.reward_instance(s,a,px,py,time_diff)*time_discount
						# time_discount_sum[s,a] += time_discount

						time_discount = self.param.lambda_r**time_diff
						R[s,a] += self.reward_instance(s,a,px,py,time_diff)
						time_discount_sum[s,a] += time_discount

			# if local update 
			else:

				customer_state = self.coordinate_to_cell_index(px,py)
				local_states = self.get_local_states(customer_state)
					
				for local_state in local_states:
					a = self.s_sp_to_a(customer_state,local_state)
					q_idx = self.sa_to_q_idx(local_state,a)

					prime_idxs = local_state*self.param.env_naction+np.arange(self.param.env_naction,dtype=int)
					reward_instance = self.reward_instance(local_state,a,px,py,time_diff)
					R[local_state,a] += reward_instance

		# if count > 0:
		# 	R /= count
		
		if count > 0:
			R /= time_discount_sum 

		# if self.param.global_reward_on:
		# 	s_idx,a_idx = np.nonzero(time_discount_sum)
		# 	R[s_idx,a_idx] /= time_discount_sum[s_idx,a_idx]

		# 	# R /= time_discount_sum

		# else:
		# 	# add noise for mdp stability 
		# 	mean_R = np.mean(np.mean(R))
		# 	R += mean_R/100 * np.random.random(R.shape)

		return R  	

	def get_MDP_P(self):
		# P in AxSxS 
		P = np.zeros((self.param.env_naction,self.param.env_ncell,self.param.env_ncell))

		for s in range(self.param.env_ncell):

			i_x,i_y = self.cell_index_to_grid_index_map[s]
			
			# 'empty' action  
			P[0,s,s] = 1.

			# 'right' action
			i_x_tp1, i_y_tp1 = (i_x+1,i_y)
			try:
				# this will fail if 
				# 	- corresponding cell index does not exist (invalid desired cell)
				# 	- outside grid index map dimensions (outside map)
				s_tp1 = self.grid_index_to_cell_index_map[i_x_tp1,i_y_tp1]
				P[1,s,s_tp1] = 1.
			except:
				P[1,s,s] = 1.

			# 'up' action
			i_x_tp1, i_y_tp1 = (i_x,i_y+1)
			try:
				s_tp1 = self.grid_index_to_cell_index_map[i_x_tp1,i_y_tp1]
				P[2,s,s_tp1] = 1.
			except:
				P[2,s,s] = 1.

			# 'left' action
			i_x_tp1, i_y_tp1 = (i_x-1,i_y)
			try:
				# this will fail if (i_x_tp1,i_y_tp1) -> cell index does not exist 
				s_tp1 = self.grid_index_to_cell_index_map[i_x_tp1,i_y_tp1]
				P[3,s,s_tp1] = 1.
			except:
				P[3,s,s] = 1.

			# 'down' action
			i_x_tp1, i_y_tp1 = (i_x,i_y-1)
			try:
				# this will fail if (i_x_tp1,i_y_tp1) -> cell index does not exist 
				s_tp1 = self.grid_index_to_cell_index_map[i_x_tp1,i_y_tp1]
				P[4,s,s_tp1] = 1.
			except:
				P[4,s,s] = 1.

		# print(P)
		# exit()

		return P  		

	def get_MDP_Pq(self,q):

		Pq = np.zeros((self.param.nq,self.param.nq))
		for s in range(self.param.env_ncell):
			for a in range(self.param.env_naction):
				q_idx = self.sa_to_q_idx(s,a)
				sp = self.get_next_state(s,a)
				ap = np.argmax(q[sp*self.param.env_naction+np.arange(self.param.env_naction,dtype=int)])
				qp_idx = self.sa_to_q_idx(sp,ap)
				Pq[q_idx,qp_idx] = 1
		self.Pq = Pq
		return Pq 

	# plotting stuff 
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

	def calc_delta_e(self,K_kp1,A_k):
		# input
		# 	- K_kp1 in [1 x 1 x ni]
		# 	- A_k in [ni x ni]
		# output: 
		# 	- delta_e, scalar 

		if not hasattr(self, 'lambda_min'):
			self.lambda_min = np.zeros((self.param.ni,self.param.sim_nt))
			self.lambda_min[:,0] = np.ones((self.param.ni))
			self.reset_timestep = 0 

		for agent_i in self.agents:
			update_lst = []
			for agent_j in self.agents:
				if A_k[agent_i.i,agent_j.i] > 0 and K_kp1[agent_j.i] > 0:
					self.lambda_min[agent_i.i,self.timestep] = 1.0 
	
		# moving average
		if True:
			if self.timestep < self.reset_timestep + self.param.htd_time_window:
				idx = np.arange(self.timestep+1)
			else:
				idx = (self.timestep-self.param.htd_time_window) + np.arange(self.param.htd_time_window+1)

			ave_lambda_min = np.mean(self.lambda_min[:,idx],axis=1)
			contraction = np.sqrt(1 - np.min(ave_lambda_min))
	
		# no moving average
		else:
			print('self.lambda_min[:,self.timestep]',self.lambda_min[:,self.timestep])
			print('np.min(self.lambda_min[:,self.timestep])',np.min(self.lambda_min[:,self.timestep]))
			contraction = np.sqrt(1 - np.min(self.lambda_min[:,self.timestep]))

		delta_e = 2*(self.param.process_noise + self.param.measurement_noise)/((1-self.param.mdp_gamma)*(1-contraction))

		if np.isnan(delta_e):
			print(1 - np.min(self.lambda_min[:,self.timestep]))			
			exit()
		
		return delta_e

	def calc_delta_d(self):
		delta_d = self.param.delta_d_ratio * np.linalg.norm(self.q)
		# delta_d = 2.5 
		return delta_d