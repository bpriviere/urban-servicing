
# standard package
import numpy as np 

# my packages
from helper_classes import Dispatch, Service, Empty
from task_assignment import centralized_linear_program, binary_log_learning


class Controller():
	def __init__(self,param,env,dispatch,task_assignment):
		self.param = param
		self.env = env

		if dispatch in ['empty']:
			self.dispatch_name = 'empty'
			self.dispatch = self.empty
		elif dispatch in ['random']:
			self.dispatch_name = 'random'
			self.dispatch = self.random
		elif dispatch in ['dtd']:
			self.dispatch_name = 'dtd'
			self.dispatch = self.dtd
		elif dispatch in ['ctd']:
			self.dispatch_name = 'ctd'
			self.dispatch = self.ctd
		elif dispatch in ['rhc']:
			self.dispatch_name = 'rhc'
			self.dispatch = self.rhc
		elif dispatch in ['bellman']:
			self.dispatch_name = 'bellman'
			self.dispatch = self.bellman

		if task_assignment in ['clp']:
			self.ta_name = 'clp'
			self.ta = centralized_linear_program
		elif task_assignment in ['blll']:
			self.ta_name = 'blll'
			self.ta = binary_log_learning

		self.name = self.dispatch_name + ' with ' + self.ta_name 
		

	# ------------ simulator -------------

	def policy(self,observation):
		
		# input: 
		# - observation is a list of customer requests, passed by reference (so you can remove customers when serviced)
		# - customer request: [time_of_request,time_to_complete,x_p,y_p,x_d,y_d], np array
		# output: 
		# - action list

		# action:
		# - if currently servicing customer: continue
		# - elif: closest agent to some 
		# - else: dispatch action chosen using some algorithm (d-td, c-td, RHC)


		# agent modes:
		# - 0: free
		# - 1: servicing

		# get free agents 
		free_agents = []
		service_agents = []
		for agent in self.env.agents:
			if agent.mode == 0: 
				free_agents.append(agent)
			elif agent.mode == 1:
				service_agents.append(agent)

		# for each customer request, assign closest available taxi
		service_assignments = []  
		time = self.param.sim_times[self.env.timestep]
		for service in observation:
			
			min_dist = np.inf
			serviced = False
			for agent in free_agents: 
				dist = np.linalg.norm([agent.x - service.x_p, agent.y - service.y_p])
				if dist < min_dist and dist < self.param.r_sense:
					min_dist = dist 
					assignment_i = agent.i 
					serviced = True

			if serviced:
				service_assignments.append((self.env.agents[assignment_i],service)) 
				observation.remove(service)
				free_agents.remove(self.env.agents[assignment_i])
			else:
				service.time_before_assignment += self.param.sim_dt

		# assign remaining idle taxis with some dispatch algorithm
		move_assignments = self.dispatch(free_agents) 

		actions = []
		for agent,service in service_assignments:
			actions.append((agent,service))
		for agent,move in move_assignments:
			actions.append((agent,move))
		for agent in service_agents:
			actions.append((agent,Empty()))

		return actions

	# ------------ dispatch -------------

	def dtd(self,agents):

		# gradient update
		print('dkif...')
		# self.dkif_ss()
		self.dkif_ms()

		# blll task assignment 
		print('ta...')
		cell_assignments = self.ta(self.env,agents)

		# assignment 
		move_assignments = self.cell_to_move_assignments(cell_assignments)

		return move_assignments

	def ctd(self,agents):
		# centralized temporal difference learning 
		
		# gradient update
		print('ckif...')
		self.ckif()

		# task assignment 
		print('ta...')
		cell_assignments = self.ta(self.env,agents)

		# assignment 
		move_assignments = self.cell_to_move_assignments(cell_assignments)

		return move_assignments 


	def rhc(self,agents):
		# receding horizon control 
		
		# no update

		# task assignment 
		print('ta...')
		cell_assignments = self.ta(self.env,agents)
		
		# assignment 
		move_assignments = self.cell_to_move_assignments(cell_assignments)
		
		return move_assignments 


	def bellman(self,agents):
		# belllman iteration update law
		
		# update
		print('bellman...')
		v,q_bellman = self.env.solve_MDP(self.env.dataset,self.param.sim_times[self.env.timestep])

		# update all agents
		for agent in self.env.agents:
			agent.q = q_bellman

		# assignment = (agent, cell) for all free agents
		print('ta...')
		cell_assignments = self.ta(self.env,agents)

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
		# transition = self.env.get_MDP_P(self.env)
		transition = self.env.P
		for agent,cell in cell_assignments:
			i = np.where(transition[cell,self.env.coordinate_to_cell_index(agent.x,agent.y),:] == 1)[0][0]
			
			if False:
				x,y = self.env.random_position_in_cell(i)
			else:
				x,y = self.env.cell_index_to_cell_coordinate(i)
				x += self.param.env_dx/2
				y += self.param.env_dy/2
			
			move_vector = np.array([x,y])
			move = Dispatch(move_vector[0],move_vector[1])
			move_assignments.append((agent,move))

		return move_assignments		


	def ckif(self):
		
		measurements = self.get_measurements()

		# get matrices 
		F = 1.0
		Q = self.param.process_noise
		R = self.param.measurement_noise
		invF = 1.0 
		invQ = 1.0/self.param.process_noise
		invR = 1.0/self.param.measurement_noise
		H = np.zeros((self.param.ni),dtype=np.float32)
		for i,(agent_i,measurement_i) in enumerate(measurements):
			if np.count_nonzero(measurement_i) > 0:
				H[i] = 1.0 

		# information transformation
		Y_km1km1 = 1/self.env.agents[0].p
		y_km1km1 = Y_km1km1 * self.env.agents[0].q

		# predict 
		M = invF * Y_km1km1 * invF
		C = M / (M + invQ)
		L = 1 - C
		Y_kkm1 = L * M * L + C * invQ * C 
		y_kkm1 = L * invF * y_km1km1 

		# innovate 
		mat_I = 0.0 #np.zeros((self.param.nq,self.param.nq))
		vec_I = np.zeros((self.param.nq))
		for agent_i, measurement_i in measurements:
			mat_I += H[agent_i.i] * invR * H[agent_i.i] 
			vec_I += H[agent_i.i] * invR * measurement_i

		# invert information transformation
		Y_kk = Y_kkm1 + mat_I
		y_kk = y_kkm1 + vec_I
		p_k = 1 / Y_kk
		q_k = p_k * y_kk
	
		for agent in self.env.agents:
			agent.q = q_k
			agent.p = p_k

	def dkif_ms(self):
		# distributed kalman information filter (measurement sharing variant)
		
		measurements = self.get_measurements()
		adjacency_matrix = self.make_adjacency_matrix()

		# init 
		q_kp1 = np.zeros((self.param.nq,self.param.ni))
		p_kp1 = np.zeros((self.param.ni))

		# get matrices v2 
		F = 1.0
		Q = self.param.process_noise
		R = self.param.measurement_noise
		invF = 1.0
		invQ = 1.0/self.param.process_noise
		invR = 1.0/self.param.measurement_noise
		H = np.zeros((self.param.ni),dtype=np.float32)
		for agent,measurement in measurements:
			if np.count_nonzero(measurement) > 0:
				H[agent.i] = 1

		for agent_i,measurement_i in measurements:

			# information transformation version 2
			Y_km1km1 = 1/agent_i.p
			y_km1km1 = Y_km1km1*agent_i.q

			# predict v2 
			M = invF * Y_km1km1 * invF
			C = M/(M+invQ)
			L = 1 - C 
			Y_kkm1 = L*M*L + C*invQ*C 
			y_kkm1 = L*invF*y_km1km1 

			# innovate v2 
			mat_I = 0.0 #np.zeros((self.param.nq))
			vec_I = np.zeros((self.param.nq))
			for agent_j, measurement_j in measurements:
				if adjacency_matrix[agent_i.i,agent_j.i] > 0:
					mat_I += H[agent_j.i] * invR
					vec_I += H[agent_j.i] * invR * measurement_j

			# invert it v2 
			Y_kk = Y_kkm1 + mat_I
			y_kk = y_kkm1 + vec_I
			P_k = 1 / Y_kk
			q_kp1[:,agent_i.i] = P_k * y_kk
			p_kp1[agent_i.i] = P_k # * np.ones((p_kp1.shape[0]))

		# assign v2
		for agent in self.env.agents:
			agent.q = q_kp1[:,agent.i]
			agent.p = p_kp1[agent.i]


	def dkif_ss(self):
		# distributed kalman information filter (state sharing variant)
		
		measurements = self.get_measurements()
		adjacency_matrix = self.make_adjacency_matrix()
		q_kp1 = np.zeros((self.param.nq,self.param.ni))
		p_kp1 = np.zeros((self.param.nq,self.param.ni))

		# get matrices
		F = np.eye(self.param.nq)
		Q = self.param.process_noise*np.eye(self.param.nq)
		R = self.param.measurement_noise*np.eye(self.param.nq)
		invF = np.eye(self.param.nq)
		invQ = 1.0/self.param.process_noise*np.eye(self.param.nq)
		invR = 1.0/self.param.measurement_noise*np.eye(self.param.nq)

		for agent_i,measurement_i in measurements:

			H = np.zeros((self.param.nq,self.param.nq))
			if np.count_nonzero(measurement_i) > 0:
				H = np.eye(self.param.nq)

			# information transformation
			Y_km1km1 = np.diag(1/agent_i.p) 
			y_km1km1 = np.dot(Y_km1km1,agent_i.q)

			# predict
			M = np.dot(invF.T, np.dot(Y_km1km1,invF))
			C = np.dot(M, np.linalg.pinv(M + invQ))
			L = np.eye(self.param.nq) - C 
			Y_kkm1 = np.dot(L,np.dot(M,L.T)) + np.dot(C,np.dot(invQ,C.T))
			y_kkm1 = np.dot(L,np.dot(invF.T,y_km1km1))

			# innovate
			mat_I = np.dot(H.T, np.dot(invR, H))
			vec_I = np.dot(H.T, np.dot(invR, measurement_i))

			# invert information transformation
			Y_kk = Y_kkm1 + mat_I
			y_kk = y_kkm1 + vec_I
			P_k = np.linalg.pinv(Y_kk)
			q_kp1[:,agent_i.i] = np.dot(P_k,y_kk)
			p_kp1[:,agent_i.i] = np.diag(P_k) 

		for agent in self.env.agents:
			agent.q = q_kp1[:,agent.i]
			agent.p = p_kp1[:,agent.i]

		for agent_i in self.env.agents:
			consensus_update = np.zeros((self.param.nq))
			for agent_j,measurement_j in measurements:
				if adjacency_matrix[agent_i.i,agent_j.i] > 0 and np.count_nonzero(measurement_j) > 0:
					consensus_update += agent_j.q-agent_i.q
			agent_i.q += consensus_update

	
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

				# if global update 
				if self.param.global_reward_on:
					for s in range(self.param.env_ncell):
						for a in range(self.param.env_naction):

							next_state = self.env.get_next_state(s,a)
							q_idx = self.env.sa_to_q_idx(s,a)
							prime_idxs = next_state*self.param.env_naction+np.arange(self.param.env_naction,dtype=int)

							reward_instance = self.env.reward_instance(s,a,px,py,time_diff)

							measurement[q_idx] += reward_instance + self.param.mdp_gamma*max(agent.q[prime_idxs]) - agent.q[q_idx]

				# if local update
				else:
					customer_state = self.env.coordinate_to_cell_index(px,py)
					local_states = self.env.get_local_states(customer_state)
					
					for local_state in local_states:
						a = self.env.s_sp_to_a(customer_state,local_state)
						q_idx = self.env.sa_to_q_idx(local_state,a)

						prime_idxs = local_state*self.param.env_naction+np.arange(self.param.env_naction,dtype=int)
						reward_instance = self.env.reward_instance(local_state,a,px,py,time_diff)
						measurement[q_idx] += reward_instance + self.param.mdp_gamma*max(agent.q[prime_idxs]) - agent.q[q_idx]

			measurements.append((agent,measurement))

		return measurements 


