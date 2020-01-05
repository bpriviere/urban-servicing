

import numpy as np 

import utilities
from agent import Dispatch, Service, Empty
from task_assignment import centralized_linear_program, binary_log_learning


class Controller():
	def __init__(self,param,env,dispatch_algorithm):
		self.param = param
		self.env = env

		if dispatch_algorithm in ['empty']:
			self.name = 'empty'
			self.dispatch = self.empty
		elif dispatch_algorithm in ['random']:
			self.name = 'random'
			self.dispatch = self.random
		elif dispatch_algorithm in ['dtd']:
			self.name = 'dtd'
			self.dispatch = self.dtd
		elif dispatch_algorithm in ['ctd']:
			self.name = 'ctd'
			self.dispatch = self.ctd
		elif dispatch_algorithm in['rhc']:
			self.name = 'rhc'
			self.dispatch = self.rhc
		else:
			exit('fatal error: param.controller_name not recognized')
		

	# ------------ simulator -------------
	def policy(self,observation):
		
		# input: 
		# - observation is a list of customer requests, passed by reference
		# - customer request: [time_of_request,time_to_complete,x_p,y_p,x_d,y_d], np array
		# output: 
		# - action list
		# - if closest and idle: agent's action is servicing a customer request
		# - elif idle: dispatch action chosen using some algorithm (d-td, c-td, RHC)
		# - else: action is continue servicing customer, or continue dispatching 


		# get available agents (on dispatch or idle)
		available_agents = []
		for agent in self.env.agents:
			if agent.mode == 0 or agent.mode == 3: 
				available_agents.append(agent)

		# get idle agents (to be dispatched)
		idle_agents = []
		for agent in self.env.agents:
			if agent.mode == 0: # idle
				idle_agents.append(agent)
		

		# for each customer request, assign closest available taxi
		service_assignments = []  
		time = self.param.sim_times[self.env.timestep]
		for service in observation:
			
			min_dist = np.inf
			serviced = False
			for agent in available_agents: 
				dist = np.linalg.norm([agent.x - service.x_p, agent.y - service.y_p])
				if dist < min_dist and dist < self.param.r_sense:
					min_dist = dist 
					assignment = agent.i 
					serviced = True

			if serviced:
				service_assignments.append((self.env.agents[assignment],service)) 
				observation.remove(service)
				available_agents.remove(self.env.agents[assignment])
			else:
				print('    not serviced: ', service)
				service.time_before_assignment += self.param.sim_dt


		# assign remaining idle customers with different dispatch algorithms 
		move_assignments = self.dispatch(idle_agents) 


		# make final actions list 
		actions = []
		for agent in self.env.agents:
			if agent in [a for a,s in service_assignments]:
				for a,s in service_assignments:
					if agent is a:
						actions.append(s)	

			elif agent in [a for a,m in move_assignments]:
				for a,m in move_assignments:
					if agent is a:
						actions.append(m)

			else: 
				empty = Empty()
				actions.append(empty)

		return actions


	# ------------ dispatch -------------

	def dtd(self,agents):

		# gradient update
		self.dkif()

		# LP task assignment 
		cell_assignments = binary_log_learning(self.env,agents)

		# assignment 
		move_assignments = self.cell_to_move_assignments(cell_assignments)

		return move_assignments

	def ctd(self,agents):
		# centralized temporal difference learning 
		
		# gradient update
		self.ckif()

		# task assignment 
		cell_assignments = binary_log_learning(self.env,agents)

		# assignment 
		move_assignments = self.cell_to_move_assignments(cell_assignments)

		return move_assignments 


	def rhc(self,agents):
		# receding horizon control 
		cell_assignments = centralized_linear_program(self.env,agents)
		move_assignments = self.cell_to_move_assignments(cell_assignments)
		return move_assignments


	def empty(self,agents):
		move_assignments = []
		for agent in agents:
			move_x = 0
			move_y = 0 
			move = Dispatch(move_x,move_y)
			move_assignments.append((agent,move))
		return move_assignments


	def random(self,agents):
		move_assignments = []
		for agent in agents:
			th = np.random.random()*2*np.pi
			move_x = np.cos(th)*self.param.taxi_speed*self.param.sim_dt
			move_y = np.sin(th)*self.param.taxi_speed*self.param.sim_dt
			move = Dispatch(move_x,move_y)
			move_assignments.append((agent,move))
		return move_assignments


	# ------------ helper fnc -------------
	def cell_to_move_assignments(self,cell_assignments):
		
		move_assignments = [] 
		transition = utilities.get_MDP_P(self.env)
		for agent,cell in cell_assignments:
			i = np.where(transition[cell,utilities.coordinate_to_cell_index(agent.x,agent.y),:] == 1)[0][0]
			
			# x,y = utilities.random_position_in_cell(i)
			x,y = utilities.cell_index_to_cell_coordinate(i)
			x += self.param.env_dx/2
			y += self.param.env_dy/2

			if self.env.param.new_on:
				move_vector = np.array([x,y])

			else: 
				move_vector = np.array([x-agent.x,y-agent.y])
				scale = self.param.taxi_speed*self.param.sim_dt/np.linalg.norm(move_vector)
				if scale < 1:
					move_vector = move_vector*scale

			move = Dispatch(move_vector[0],move_vector[1])
			move_assignments.append((agent,move))

		return move_assignments		


	def ckif(self):
		
		measurements = self.get_measurements()

		measurement_update = np.zeros(self.param.nq)
		for agent_i,measurement_i in measurements:
			for k,measurement_ik in enumerate(measurement_i):
				if not measurement_ik==0:
					alpha = self.get_learning_rate(agent_i,k)
					measurement_update[k] += alpha*measurement_i[k]

		for a in self.env.agents:
			a.q = a.q + measurement_update 



	def dkif(self):
		
		measurements = self.get_measurements()
		adjacency_matrix = self.make_adjacency_matrix()
		q_tp1 = np.zeros((self.param.nq,self.param.ni))

		for agent_i,measurement_i in measurements:

			measurement_update = np.zeros(self.param.nq)
			consensus_update = np.zeros((self.param.nq))
			for k,measurement_ik in enumerate(measurement_i):
				if not measurement_ik==0:
					# measurement update
					alpha_i = self.get_learning_rate(agent_i,k)
					measurement_update[k] = alpha_i*measurement_ik

					# consensus update 
					# for agent_j,measurement_j in measurements:
					# 	for l,measurement_jl in enumerate(measurement_i):
					# 		if not measurement_jl==0:
					# 			consensus_update[l] += adjacency_matrix[agent_i.i,agent_j.i]*(agent_j.q[l]-agent_i.q[l])

			q_tp1[:,agent_i.i] = agent_i.q + measurement_update + consensus_update
		
		for agent in self.env.agents:
			agent.q = q_tp1[:,agent.i]


	def get_learning_rate(self,agent,q_idx):

		# predict
		p_kkm1 = agent.p + self.param.process_noise

		# innovate 
		s = p_kkm1 + self.param.measurement_noise 

		# gain
		alpha = p_kkm1/s 

		# update agent covariance
		agent.p = (1-alpha)*p_kkm1 

		print('alpha: ', alpha)
		print('agent.p ', agent.p)

		return alpha 

	def make_adjacency_matrix(self):

		# A = np.zeros((self.param.ni,self.param.ni))
		# for agent_i in self.env.agents:
		# 	p_i = np.array([agent_i.x,agent_i.y])
		# 	for agent_j in self.env.agents:
		# 		if not agent_i is agent_j:
		# 			p_j = np.array([agent_j.x,agent_j.y])
		# 			dist = np.linalg.norm(p_i-p_j)
		# 			if dist < self.param.r_comm:
		# 				A[agent_i.i,agent_j.i] = 1

		A = np.ones((self.param.ni,self.param.ni))

		# for agent_i in self.env.agents:
		# 	i_nn = sum(A[agent_i.i,:])
		# 	for agent_j in self.env.agents:
		# 		j_nn = sum(A[agent_j.i,:])
		# for i in range(self.param.ni):
		# 	A[i,:] = A[i,:]/sum(A[i,:])

		# 	print('A[i,:]:',A[i,:])
		# 	print('A[:,i]:',A[:,i])
		# exit()

		A = A/self.param.ni
		return A

	# def get_measurement_model(self,agent,measurements):
	# 	for a,m in measurements:
	# 		if agent is a:
	# 			break
	# 	h_diag = [1 if not m_i==0 else 0 for m_i in m] 
	# 	h_i = np.diag(h_diag)
		
	# 	print('h_i: ',h_i)
	# 	print('m: ',m)
	# 	return h_i 

	def get_measurements(self):
		measurements = []
		for agent in self.env.agents:
			
			measurement = np.zeros(self.param.nq)

			if agent.update:
				agent.update = False

				px = agent.service.x_p
				py = agent.service.y_p
				time_of_request = agent.service.time

				global_state_update = False
				if global_state_update:
					for s in range(self.param.env_ncell):
						for a in range(self.param.env_naction):
							next_state = utilities.get_next_state(self.env,s,a)
							next_state_x,next_state_y = utilities.cell_index_to_cell_coordinate(next_state)
							wait_cost = self.env.eta(px,py,next_state_x,next_state_y,time_of_request)
							action_cost = self.param.lambda_a*(not a==0)
							reward_instance = -1*(wait_cost+action_cost)

							q_idx = utilities.sa_to_q_idx(s,a)
							prime_idxs = next_state*self.param.env_naction+np.arange(self.param.env_naction,dtype=int)
							measurement[q_idx] = reward_instance + self.param.mdp_gamma*max(agent.q[prime_idxs]) - agent.q[q_idx]

				else:

					state = utilities.coordinate_to_cell_index(agent.x,agent.y)
					prev_sa_lst = utilities.get_prev_sa_lst(self.env,state)

					for prev_s,prev_a in prev_sa_lst:
						prev_sx,prev_sy = utilities.cell_index_to_cell_coordinate(prev_s)
						q_idx = utilities.sa_to_q_idx(prev_s,prev_a)

						# this method requires knowledge of the ETA model, whereas previously we only had samples from it.... 
						wait_cost = self.env.eta(px,py,prev_sx,prev_sy,time_of_request)
						action_cost = self.param.lambda_a*(not prev_a==0)
						reward_instance = -1*(wait_cost+action_cost)

						prime_idxs = state*self.param.env_naction+np.arange(self.param.env_naction,dtype=int)
						prev_idx = prev_s*self.param.env_naction + prev_a

						measurement[q_idx] = reward_instance + self.param.mdp_gamma*max(agent.q[prime_idxs]) - agent.q[prev_idx]

			measurements.append((agent,measurement))

		return measurements 


