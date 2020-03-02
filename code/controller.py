

import numpy as np 

import utilities
from helper_classes import Dispatch, Service, Empty
# from agent import Dispatch, Service, Empty
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
		elif dispatch_algorithm in ['rhc']:
			self.name = 'rhc'
			self.dispatch = self.rhc
		elif dispatch_algorithm in ['bellman']:
			self.name = 'bellman'
			self.dispatch = self.bellman		

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


		# get available agents (on dispatch or idle or pickup)
		available_agents = []
		for agent in self.env.agents:
			if agent.mode == 0 or agent.mode == 3 or agent.mode == 1: 
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
				# print('    not serviced: ', service)
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
		# no update
		# task assignment 
		cell_assignments = binary_log_learning(self.env,agents)
		# assignment 
		move_assignments = self.cell_to_move_assignments(cell_assignments)
		return move_assignments 


	def bellman(self,agents):
		# belllman iteration update law
		
		# update
		v,q_bellman = utilities.solve_MDP(self.env, self.env.dataset,self.param.sim_times[self.env.timestep])

		# update all agents
		for agent in self.env.agents:
			agent.q = q_bellman

		# assignment = (agent, cell) for all free agents
		cell_assignments = binary_log_learning(self.env,agents)

		# assignment = (agent, action) for all agents in cell assignments
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
			
			if False:
				x,y = utilities.random_position_in_cell(i)
			else:
				x,y = utilities.cell_index_to_cell_coordinate(i)
				x += self.param.env_dx/2
				y += self.param.env_dy/2
			
			move_vector = np.array([x,y])
			move = Dispatch(move_vector[0],move_vector[1])
			move_assignments.append((agent,move))

		return move_assignments		


	def ckif(self):
		
		measurements = self.get_measurements()

		# get matrices
		F = np.eye(self.param.nq)
		Q = self.param.process_noise*np.eye(self.param.nq)
		R = self.param.measurement_noise*np.eye(self.param.nq)
		invF = np.eye(self.param.nq)
		invQ = 1.0/self.param.process_noise*np.eye(self.param.nq)
		invR = 1.0/self.param.measurement_noise*np.eye(self.param.nq)

		H = np.zeros((self.param.nq,self.param.nq,self.param.ni),dtype=np.float32)
		for i,(agent_i,measurement_i) in enumerate(measurements):
			if np.count_nonzero(measurement_i) > 0:
				H[:,:,i] = np.eye(self.param.nq)

		# information transformation
		Y_km1km1 = 1/self.env.agents[0].p * np.eye(self.param.nq) 
		y_km1km1 = np.dot(Y_km1km1,self.env.agents[0].q)

		# predict
		M = np.dot(invF.T, np.dot(Y_km1km1,invF))
		C = np.dot(M, np.linalg.pinv(M + invQ))
		L = np.eye(self.param.nq) - C 
		Y_kkm1 = np.dot(L,np.dot(M,L.T)) + np.dot(C,np.dot(invQ,C.T))
		y_kkm1 = np.dot(L,np.dot(invF.T,y_km1km1))

		# innovate
		sum_I = np.zeros((self.param.nq,self.param.nq))
		sum_i = np.zeros((self.param.nq))
		for i, (agent_i, measurement_i) in enumerate(measurements):
			sum_I += np.dot(H[:,:,i].T, np.dot(invR, H[:,:,i]))
			sum_i += np.dot(H[:,:,i].T, np.dot(invR, measurement_i))

		# invert information transformation
		Y_kk = Y_kkm1 + sum_I
		y_kk = y_kkm1 + sum_i
		P_k = np.linalg.pinv(Y_kk)
		q_k = np.dot(P_k,y_kk)
	
		for agent in self.env.agents:
			agent.q = q_k
			agent.p = P_k[0][0]


	def dkif(self):
		
		measurements = self.get_measurements()
		adjacency_matrix = self.make_adjacency_matrix()
		q_kp1 = np.zeros((self.param.nq,self.param.ni))
		p_kp1 = np.zeros((self.param.ni))

		# get matrices
		F = np.eye(self.param.nq)
		Q = self.param.process_noise*np.eye(self.param.nq)
		R = self.param.measurement_noise*np.eye(self.param.nq)
		invF = np.eye(self.param.nq)
		invQ = 1.0/self.param.process_noise*np.eye(self.param.nq)
		invR = 1.0/self.param.measurement_noise*np.eye(self.param.nq)

		for i, (agent_i,measurement_i) in enumerate(measurements):

			H = np.zeros((self.param.nq,self.param.nq))
			if np.count_nonzero(measurement_i) > 0:
				H = np.eye(self.param.nq)

			# information transformation
			Y_km1km1 = 1/self.env.agents[0].p * np.eye(self.param.nq) 
			y_km1km1 = np.dot(Y_km1km1,self.env.agents[0].q)

			# predict
			M = np.dot(invF.T, np.dot(Y_km1km1,invF))
			C = np.dot(M, np.linalg.pinv(M + invQ))
			L = np.eye(self.param.nq) - C 
			Y_kkm1 = np.dot(L,np.dot(M,L.T)) + np.dot(C,np.dot(invQ,C.T))
			y_kkm1 = np.dot(L,np.dot(invF.T,y_km1km1))

			# innovate
			mat_I = np.dot(H.T, np.dot(invR, H))
			vec_i = np.dot(H.T, np.dot(invR, measurement_i))

			# add consensus term
			consensus_update = np.zeros((self.param.nq))
			for agent_j,measurement_j in measurements:
				if np.count_nonzero(measurement_j) > 0:
					consensus_update += adjacency_matrix[agent_i.i,agent_j.i]*(agent_j.q-agent_i.q)

			# invert information transformation
			Y_kk = Y_kkm1 + mat_I
			y_kk = y_kkm1 + vec_i
			P_k = np.linalg.pinv(Y_kk)
			q_kp1[:,i] = np.dot(P_k,y_kk) + consensus_update
			p_kp1[i] = P_k[0][0]

		for i,agent in enumerate(self.env.agents):
			agent.q = q_kp1[:,i]
			agent.p = p_kp1[i]

	def make_adjacency_matrix(self):

		A = np.zeros((self.param.ni,self.param.ni))
		for agent_i in self.env.agents:
			p_i = np.array([agent_i.x,agent_i.y])
			for agent_j in self.env.agents:
				if not agent_i is agent_j:
					p_j = np.array([agent_j.x,agent_j.y])
					dist = np.linalg.norm(p_i-p_j)
					if dist < self.param.r_comm:
						A[agent_i.i,agent_j.i] = 1

		A = A/self.param.ni
		return A

	def get_measurements(self):
		measurements = []
		for agent in self.env.agents:
			
			measurement = np.zeros(self.param.nq)

			if agent.update:
				agent.update = False

				px = agent.service.x_p
				py = agent.service.y_p
				time_of_request = agent.service.time
				time_diff = self.param.sim_times[self.env.timestep] - time_of_request

				global_state_update = True
				if global_state_update:
					for s in range(self.param.env_ncell):
						for a in range(self.param.env_naction):

							next_state = utilities.get_next_state(self.env,s,a)
							q_idx = utilities.sa_to_q_idx(s,a)
							prime_idxs = next_state*self.param.env_naction+np.arange(self.param.env_naction,dtype=int)

							reward_instance = utilities.reward_instance(self.env,s,a,px,py)

							measurement[q_idx] += reward_instance + self.param.mdp_gamma*max(agent.q[prime_idxs]) - agent.q[q_idx]

				# else:

				# 	state = utilities.coordinate_to_cell_index(agent.x,agent.y)
				# 	prev_sa_lst = utilities.get_prev_sa_lst(self.env,state)

				# 	for prev_s,prev_a in prev_sa_lst:
				# 		prev_sx,prev_sy = utilities.cell_index_to_cell_coordinate(prev_s)
				# 		q_idx = utilities.sa_to_q_idx(prev_s,prev_a)

				# 		reward_instance = utilities.reward_instance(self.env,prev_s,prev_a,px,py)

				# 		prime_idxs = state*self.param.env_naction+np.arange(self.param.env_naction,dtype=int)
				# 		prev_idx = prev_s*self.param.env_naction + prev_a

				# 		measurement[q_idx] += reward_instance + self.param.mdp_gamma*max(agent.q[prime_idxs]) - agent.q[prev_idx]

			measurements.append((agent,measurement))

		return measurements 


