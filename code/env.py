
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
		self.waiting_customer_locations = dict()

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

			px = customer_requests[i,2]
			py = customer_requests[i,3]
			dx = customer_requests[i,4]
			dy = customer_requests[i,5]

			# check if points are in map 
			if self.check_in_map(px,py) and self.check_in_map(dx,dy):

				pickup_state = self.coordinate_to_cell_index(px,py)
				dropoff_state = self.coordinate_to_cell_index(dx,dy)
				
				if pickup_state in self.valid_cells and dropoff_state in self.valid_cells:
					service = Service(customer_requests[i,:])
					self.observation.append(service)
					self.waiting_customer_locations[service] = (service.x_p,service.y_p)

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

		state = self.extract_state()
		
		# increment timestep
		self.timestep += 1
		return reward, state

	def extract_state(self):

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

		# save customer times 
		waiting_customers_location = []
		for service, location in self.waiting_customer_locations.items():
			waiting_customers_location.append(location)

		# - distributions 
		# gmm_distribution = self.get_curr_im_gmm()
		agents_distribution = self.get_curr_im_agents()
		free_agents_distribution = self.get_curr_im_free_agents()
		agents_value_fnc_distribution = self.get_curr_im_value()
		# agents_ave_vec_action_distribution = self.get_curr_ave_vec_action(agents_location, agents_vec_action)

		# put desired numpy arrays into dictionary
		state = dict()
		for state_key in self.param.state_keys:
			state[state_key] = eval(state_key)

		return state


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
			agent.pickup_vector = agent.pickup_vector/agent.pickup_dist
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
					self.waiting_customer_locations.pop(agent.service, None) 
				
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

	def local_boltzmann_policy(self,q,state_action_pairs):
		# which action should agent take 

		# get q(s,a) indices of where taxi could go		
		idxs = self.sa_to_q_idxs(state_action_pairs)

		# normalize q first for numerical stability 
		q = (q[idxs] - min(q[idxs]))/(max(q[idxs])-min(q[idxs]))

		# boltzman eq # check dimensionality of this thing 
		pi = np.exp(self.param.ta_beta*q) / sum(np.exp(self.param.ta_beta*(q)))
		
		return pi

	def global_boltzmann_policy(self,q):
		# which action should agent take 

		# normalize q first for numerical stability 
		q = (q - min(q))/(max(q)-min(q))

		# boltzman eq 
		pi = np.exp(self.param.ta_beta*q) / sum(np.exp(self.param.ta_beta*(q)))
		
		return pi


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

	def sa_to_q_idxs(self,state_action_pairs):
		q_idxs = []
		for s,a in state_action_pairs:
			q_idxs.append(self.sa_to_q_idx(s,a))
		return q_idxs

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

	def get_local_transitions(self,s):
		local_states = []
		local_actions = []
		for a in range(self.param.env_naction):
			next_state = self.get_next_state(s,a) 
			if not next_state in local_states:
				local_states.append(next_state)
				local_actions.append(a)
		return local_states, local_actions

	# mdp stuff 
	def solve_MDP(self,dataset,curr_time):

		# only consider until curr_time
		eval_dataset = dataset[dataset[:,0] <= curr_time,:]

		if eval_dataset.shape[0] > self.param.mdp_max_data:
			eval_dataset = eval_dataset[-self.param.mdp_max_data:,:]

		print('MDP eval_dataset.shape: ',eval_dataset.shape)

		P = self.get_MDP_P() # in AxSxS
		
		self.P = P
		
		R = self.get_MDP_R(eval_dataset,curr_time) # in SxA

		mdp = ValueIteration(P,R,self.param.mdp_gamma,self.param.mdp_eps,self.param.mdp_max_iter)
		mdp.run()
		V = np.array(mdp.V)
		Q = self.get_MDP_Q(R,V,self.param.mdp_gamma)

		self.v = V 
		self.q = Q 
		self.r = R.flatten()
		return self.v,self.q,self.r


	def get_customer_demand(self,dataset,curr_time):

		# only consider until curr_time
		eval_dataset = dataset[dataset[:,0] <= curr_time,:]

		if eval_dataset.shape[0] > self.param.mdp_max_data:
			eval_dataset = eval_dataset[-self.param.mdp_max_data:,:]

		print('customer demand eval_dataset.shape: ',eval_dataset.shape)

		w = np.zeros((self.param.env_ncell))
		for customer_request in eval_dataset:

			px = customer_request[2]
			py = customer_request[3]
			dx = customer_request[4]
			dy = customer_request[5]

			# check if points are in map 
			if self.check_in_map(px,py) and self.check_in_map(dx,dy):

				pickup_state = self.coordinate_to_cell_index(px,py)
				dropoff_state = self.coordinate_to_cell_index(dx,dy)
				
				if pickup_state in self.valid_cells and dropoff_state in self.valid_cells:
					w[pickup_state] += 1
		w /= sum(w) 
		return w 


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

		idx = dataset[:,0] < curr_time
		dataset = dataset[idx,:]
		ndata = dataset.shape[0]
		
		R = np.zeros((self.param.env_ncell,self.param.env_naction))
		if ndata > 0 :

			tor = dataset[:,0]
			px = dataset[:,2]
			py = dataset[:,3] 

			time_discount = self.param.lambda_r**np.maximum((curr_time-tor)/self.param.sim_dt,1)

			for s in range(self.param.env_ncell):
				
				sx,sy = self.cell_index_to_cell_coordinate(s)
				sx += self.param.env_dx/2
				sy += self.param.env_dy/2

				for a in range(self.param.env_naction):

					sp = self.get_next_state(s,a)
					spx,spy = self.cell_index_to_cell_coordinate(sp)
					spx += self.param.env_dx/2
					spy += self.param.env_dy/2

					time_s_to_sp = self.eta(sx,sy,spx,spy)
					time_sp_to_c = np.linalg.norm((px-spx,py-spy),axis=0) / self.param.taxi_speed 

					reward = 1/(time_s_to_sp + time_sp_to_c)
					
					R[s,a] = sum(reward * time_discount)

			R /= sum(time_discount)

		return R

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

		return reward 


	def get_MDP_P(self):
		# P in AxSxS 
		P = np.zeros((self.param.env_naction,self.param.env_ncell,self.param.env_ncell))

		for s in range(self.param.env_ncell):

			i_x,i_y = self.cell_index_to_grid_index_map[s]

			for a in range(self.param.env_naction):
				if a == 0:
					# 'stay' 
					i_x_tp1, i_y_tp1 = (i_x,i_y)
				elif a == 1:
					# 'right'
					i_x_tp1, i_y_tp1 = (i_x+1,i_y)
				elif a == 2:
					# 'top'
					i_x_tp1, i_y_tp1 = (i_x,i_y+1)
				elif a == 3:
					# 'left'
					i_x_tp1, i_y_tp1 = (i_x-1,i_y)		
				elif a == 4:
					# 'down'
					i_x_tp1, i_y_tp1 = (i_x,i_y-1)													

				
				# if inside grid map dimensions 
				if i_x_tp1 < len(self.param.env_x) and i_x_tp1 >= 0 and i_y_tp1 < len(self.param.env_y) and i_y_tp1 >= 0:
					s_tp1 = self.grid_index_to_cell_index_map[i_x_tp1,i_y_tp1]

					if s_tp1 in self.valid_cells:
						P[a,s,s_tp1] = 1.
					else:
						P[a,s,s] = 1.
				else:
					P[a,s,s] = 1.

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
			value_fnc_i = self.boltzmann_policy(agent.q)
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

	def calc_delta_e(self,A_k):
		# input
		# 	- K_kp1 in [1 x 1 x ni]
		# 	- A_k in [ni x ni]
		# output: 
		# 	- delta_e, scalar 

		if not hasattr(self, 'lambda_min'):
			self.lambda_min = np.zeros((self.param.ni,self.param.sim_nt))
			self.lambda_min[:,0] = np.ones((self.param.ni))
			self.reset_timestep = 0 

		if self.timestep == self.reset_timestep:
			delta_e = 0 
		else:

			self.lambda_min[:,self.timestep] = np.sum(A_k,axis=1)
		
			# moving average
			if self.timestep < self.reset_timestep + self.param.htd_time_window:
				idx = self.reset_timestep + np.arange(self.timestep - self.reset_timestep)
			else:
				idx = self.timestep - self.param.htd_time_window + np.arange(self.param.htd_time_window)

			ave_lambda_min = np.mean(self.lambda_min[:,idx],axis=1)

			# delta_e = 2*(self.param.process_noise + self.param.measurement_noise)/ \
			# 	((1-self.param.mdp_gamma)*(1-np.sqrt(1 - np.min(ave_lambda_min))))
			# delta_e = 2*(self.param.process_noise + self.param.measurement_noise)/ \
			# 	((1-self.param.mdp_gamma)*(1-np.sqrt(1 - np.mean(ave_lambda_min))))

			delta_e = 2 * np.sqrt ( self.param.nq * (self.param.process_noise + self.param.measurement_noise))/ \
				((1-self.param.mdp_gamma)*(1-np.sqrt(1 - np.min(ave_lambda_min))))	

		if np.isnan(delta_e):
			print('delta_e: ', delta_e)
			exit()
		
		return delta_e

	def calc_delta_d(self):
		# note this is env.q which is the result of solving the mdp 
		delta_d = self.param.delta_d_ratio * np.linalg.norm(self.q) * self.param.nq
		# delta_d = self.param.delta_d_ratio * np.linalg.norm(self.q)
		# delta_d = 2.5 
		return delta_d

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

	# Utility Stuff
		# 'cell_index' : element of [0,...,env_ncell]
		# 'cell_coordinate' : (x,y) coordinates of bottom left corner of cell 
		# 'xy_cell_index' : (i_x,i_y) indices corresponding to elements of env_x, env_y
		# 'coordinate' : free (x,y) coordinate, not constrained by being on gridlines

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

	def check_valid(self,x,y):
		i = self.coordinate_to_cell_index(x,y)
		return i in self.valid_cells

	def check_in_map(self,x,y):
		return x >= self.valid_xmin and x <= self.valid_xmax and y >= self.valid_ymin and y <= self.valid_ymax
