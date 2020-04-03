
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
		elif dispatch in ['htd']:
			self.dispatch_name = 'htd'
			self.dispatch = self.htd
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

	def dtd(self,free_agents):
		# distributed temporal difference

		# get stuff 
		r_k = np.zeros((self.param.nq,self.param.ni))
		q_k = np.zeros((self.param.nq,self.param.ni))
		p_k = np.zeros((1,self.param.ni))
		Pq_k = np.zeros((self.param.nq,self.param.nq,self.param.ni))
		q_kp1 = np.zeros((self.param.nq,self.param.ni))
		r_kp1 = np.zeros((self.param.nq,self.param.ni))
		for agent in self.env.agents:
			r_k[:,agent.i] = agent.r 
			q_k[:,agent.i] = agent.q 
			p_k[:,agent.i] = agent.p 
			Pq_k[:,:,agent.i] = self.env.get_MDP_Pq(agent.q)

		# measurements 
		z_kp1,H_kp1 = self.get_measurements()

		# get adjacency matrix 
		A_k = self.make_adjacency_matrix()

		# dtd 
		print('dkif...')
		r_kp1,p_kp1,K_kp1 = self.dkif(r_k,p_k,z_kp1,H_kp1)

		# temporal difference
		alpha = self.param.td_alpha
		for agent in self.env.agents:
			td_error = r_kp1[:,agent.i]+self.param.mdp_gamma*np.dot(Pq_k[:,:,agent.i],q_k[:,agent.i])-q_k[:,agent.i]
			q_kp1[:,agent.i] = q_k[:,agent.i] + alpha*td_error

		# temporal difference
		alpha = self.param.td_alpha
		for agent in self.env.agents:
			td_error = r_kp1[:,agent.i]+self.param.mdp_gamma*np.dot(Pq_k[:,:,agent.i],q_k[:,agent.i])-q_k[:,agent.i]
			q_kp1[:,agent.i] = q_k[:,agent.i] + alpha*td_error

		# old / buggy 
		# # kalman gain 
		# p_kp1,K_kp1 = self.kalman(p_k,H_kp1,A_k)

		# # reward estimation 
		# for agent_i in self.env.agents:
		# 	update_i = np.zeros((self.param.nq))
		# 	for agent_j in self.env.agents:
		# 		if A_k[agent_i.i,agent_j.i] > 0:
		# 			update_i += (K_kp1[:,:,agent_j.i] * H_kp1[:,agent_j.i] * (z_kp1[:,agent_j.i] - r_k[:,agent_i.i])).squeeze()
		# 	r_kp1[:,agent_i.i] = r_k[:,agent_i.i] + update_i 

		# update agents
		for agent in self.env.agents:
			agent.r = r_kp1[:,agent.i]
			agent.p = p_kp1[:,agent.i]
			agent.q = q_kp1[:,agent.i]

		# task assignment 
		print('ta...')
		cell_assignments = self.ta(self.env,free_agents)

		# assignment 
		move_assignments = self.cell_to_move_assignments(cell_assignments)

		return move_assignments 


	def htd(self,free_agents):
		# hybrid temporal difference

		# get stuff 
		r_k = np.zeros((self.param.nq,self.param.ni))
		q_k = np.zeros((self.param.nq,self.param.ni))
		p_k = np.zeros((1,self.param.ni))
		Pq_k = np.zeros((self.param.nq,self.param.nq,self.param.ni))
		q_kp1 = np.zeros((self.param.nq,self.param.ni))
		r_kp1 = np.zeros((self.param.nq,self.param.ni))
		for agent in self.env.agents:
			r_k[:,agent.i] = agent.r 
			q_k[:,agent.i] = agent.q 
			p_k[:,agent.i] = agent.p 
			Pq_k[:,:,agent.i] = self.env.get_MDP_Pq(agent.q)

		# measurements 
		z_kp1,H_kp1 = self.get_measurements()

		# get adjacency matrix 
		A_k = self.make_adjacency_matrix()

		# kalman gain 
		# p_kp1,K_kp1 = self.kalman(p_k,H_kp1,A_k)

		# dtd 
		print('dkif...')
		r_kp1,p_kp1,K_kp1 = self.dkif(r_k,p_k,z_kp1,H_kp1)

		# temporal difference
		alpha = self.param.td_alpha
		for agent in self.env.agents:
			td_error = r_kp1[:,agent.i]+self.param.mdp_gamma*np.dot(Pq_k[:,:,agent.i],q_k[:,agent.i])-q_k[:,agent.i]
			q_kp1[:,agent.i] = q_k[:,agent.i] + alpha*td_error

		# error
		delta_e = self.env.calc_delta_e(K_kp1,A_k)
		delta_d = self.env.calc_delta_d()

		print('delta_e: ', delta_e)
		print('delta_d: ', delta_d)
		if delta_e > delta_d and self.env.timestep > self.env.reset_timestep + self.param.htd_time_window: 
			# bellman
			print('bellman...')
			self.env.reset_timestep = self.env.timestep
			v,q,r = self.env.solve_MDP(self.env.dataset,self.param.sim_times[self.env.timestep])

			for agent in self.env.agents:
				r_kp1[:,agent.i] = r 
				q_kp1[:,agent.i] = q
				p_kp1[:,agent.i] = self.param.p0 

		# update agents
		for agent in self.env.agents:
			agent.r = r_kp1[:,agent.i]
			agent.p = p_kp1[:,agent.i]
			agent.q = q_kp1[:,agent.i]

		# task assignment 
		print('ta...')
		cell_assignments = self.ta(self.env,free_agents)

		# assignment 
		move_assignments = self.cell_to_move_assignments(cell_assignments)

		return move_assignments 		

	def ctd(self,free_agents):
		# centralized temporal difference

		# get stuff 
		r_k = np.zeros((self.param.nq,self.param.ni))
		q_k = np.zeros((self.param.nq,self.param.ni))
		p_k = np.zeros((1,self.param.ni))
		Pq_k = np.zeros((self.param.nq,self.param.nq,self.param.ni))
		q_kp1 = np.zeros((self.param.nq,self.param.ni))
		r_kp1 = np.zeros((self.param.nq,self.param.ni))
		for agent in self.env.agents:
			r_k[:,agent.i] = agent.r 
			q_k[:,agent.i] = agent.q 
			p_k[:,agent.i] = agent.p 
			Pq_k[:,:,agent.i] = self.env.get_MDP_Pq(agent.q)

		# measurements 
		z_kp1,H_kp1 = self.get_measurements()

		# estimate reward
		print('ckif...')
		r_kp1,p_kp1,K_kp1 = self.ckif(r_k,p_k,z_kp1,H_kp1)

		# temporal difference
		alpha = self.param.td_alpha
		for agent in self.env.agents:
			# td_error = r_kp1[:,agent.i]+self.param.mdp_gamma*np.dot(Pq_k[:,:,agent.i],q_k[:,agent.i])-q_k[:,agent.i]
			td_error = r_kp1+self.param.mdp_gamma*np.dot(Pq_k[:,:,agent.i],q_k[:,agent.i])-q_k[:,agent.i]
			q_kp1[:,agent.i] = q_k[:,agent.i] + alpha*td_error

		# update agents
		for agent in self.env.agents:
			# agent.r = r_kp1[:,agent.i]
			agent.r = r_kp1
			agent.p = p_kp1
			agent.q = q_kp1[:,agent.i]

		# task assignment 
		print('ta...')
		cell_assignments = self.ta(self.env,free_agents)

		# assignment 
		move_assignments = self.cell_to_move_assignments(cell_assignments)

		return move_assignments 

	def rhc(self,agents):
		# receding horizon control 
		
		# no update
		print('rhc...')

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
		v,q,r = self.env.solve_MDP(self.env.dataset,self.param.sim_times[self.env.timestep])

		# update all agents
		for agent in self.env.agents:
			agent.q = q

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

	# estimation methods 
	def ckif(self,r_k,p_k,z_kp1,H_kp1):
		# centralized kalman information filter to estimate reward, r
		# input
		# 	- r_k : reward at previous timestep, numpy in nq x ni
		# 	- p_k : covariance at previous timestep, numpy in nq x ni
		# 	- z_kp1 : measurement, numpy in nq x ni 
		# 	- H_kp1 : measurement model, numpy in 1 x ni 
		# output
		# 	- r_kp1 : next estimate, numpy in nq x 1 
		# 	- p_kp1 : next covariance, numpy in nq x 1 

		# init 
		r_kp1 = np.zeros((self.param.nq))
		p_kp1 = np.zeros((1,self.param.nq))
		K_kp1 = np.zeros((self.param.ni))
		
		# get matrices 
		F = 1.0
		Q = self.param.process_noise
		R = self.param.measurement_noise
		invF = 1.0 
		invQ = 1.0/self.param.process_noise
		invR = 1.0/self.param.measurement_noise

		# ckif 

		# all agents have same info 
		p_k = p_k[:,0]
		r_k = r_k[:,0]

		# information transformation
		Y_kk = 1/p_k
		y_kk = Y_kk*r_k

		# predict 
		M = invF * Y_kk * invF
		C = M / (M + invQ)
		L = 1 - C
		Y_kp1k = L * M * L + C * invQ * C 
		y_kp1k = L * invF * y_kk 

		# innovate 
		mat_I = 0.0 # np.shape(H_kp1[0]) 
		vec_I = np.zeros((self.param.nq))
		for agent in self.env.agents: 
			measurement = z_kp1[:,agent.i] 
			measurement_model = H_kp1[:,agent.i] 
			mat_I += measurement_model * invR * measurement_model 
			vec_I += measurement_model * invR * measurement		

			# get learning rate
			S = measurement_model * Y_kp1k * measurement_model + R
			invS = 1/S
			K_kp1[agent.i] = Y_kp1k * measurement_model * invS 

		# invert information transformation
		Y_kp1kp1 = Y_kp1k + mat_I
		y_kp1kp1 = y_kp1k + vec_I
		p_kp1 = 1 / Y_kp1kp1
		r_kp1 = p_kp1 * y_kp1kp1 

		return r_kp1,p_kp1,K_kp1


	# def kalman(self,p_k,H_kp1,A_k):
	# 	# kalman gain for distributed 
	# 	# input
	# 	# 	- p_k : covariance at previous timestep, numpy in 1 x ni
	# 	# 	- H_kp1 : measurement model, numpy in 1 x ni 
	# 	# output
	# 	# 	- p_kp1 : next covariance, numpy in nq x ni 
	# 	# 	- K_kp1 : kalman gain, numpy in 1 x 1 x ni

	# 	measurements = self.get_measurements()

	# 	# init 
	# 	K_kp1 = np.zeros((self.param.ni))
	# 	p_kp1 = np.zeros((1,self.param.ni))
		
	# 	# get matrices 
	# 	F = 1.0
	# 	Q = self.param.process_noise
	# 	R = self.param.measurement_noise
	# 	invF = 1.0 
	# 	invQ = 1.0/Q
	# 	invR = 1.0/R

	# 	# kalman gain information filter method  
	# 	for agent_i in self.env.agents:

	# 		# information transformation
	# 		Y_kk = 1/p_k[:,agent_i.i]

	# 		# predict 
	# 		M = invF * Y_kk * invF
	# 		C = M / (M + invQ)
	# 		L = 1 - C
	# 		Y_kp1k = L*M*L + C*invQ*C 

	# 		# innovate 
	# 		mat_I = 0.0 # np.shape(H_kp1[0]) 
	# 		for agent_j in self.env.agents: 
	# 			if A_k[agent_i.i,agent_j.i] > 0:
	# 				measurement_model = H_kp1[:,agent_j.i] 
	# 				mat_I += measurement_model * invR * measurement_model 
	
	# 		# kalman gain  
	# 		S = H_kp1[:,agent_i.i] * Y_kp1k * H_kp1[:,agent_i.i] + R
	# 		invS = 1/S
	# 		K_kp1[agent_i.i] = Y_kp1k * H_kp1[:,agent_i.i] * invS 		

	# 		# invert information transformation
	# 		Y_kp1kp1 = Y_kp1k + mat_I
	# 		p_kp1[:,agent_i.i] = 1 / Y_kp1kp1

	# 	return p_kp1,K_kp1


	def dkif(self,r_k,p_k,z_kp1,H_kp1):
		# distributed kalman information filter to estimate reward, r
		# input
		# 	- r_k : reward at previous timestep, numpy in nq x ni
		# 	- p_k : covariance at previous timestep, numpy in 1 x ni
		# 	- z_kp1 : measurement, numpy in nq x ni 
		# 	- H_kp1 : measurement model, numpy in 1 x ni 
		# output
		# 	- r_kp1 : next estimate, numpy in nq x ni
		# 	- p_kp1 : next covariance, numpy in nq x ni 

		measurements = self.get_measurements()
		adjacency_matrix = self.make_adjacency_matrix()

		# init 
		r_kp1 = np.zeros((self.param.nq,self.param.ni))
		p_kp1 = np.zeros((1,self.param.ni))
		K_kp1 = np.zeros((self.param.ni))
		
		# get matrices 
		F = 1.0
		Q = self.param.process_noise
		R = self.param.measurement_noise
		invF = 1.0 
		invQ = 1.0/self.param.process_noise
		invR = 1.0/self.param.measurement_noise

		# dkif 
		for agent_i in self.env.agents:

			# information transformation
			Y_kk = 1/p_k[:,agent_i.i]
			y_kk = Y_kk*r_k[:,agent_i.i]

			# predict 
			M = invF * Y_kk * invF
			C = M / (M + invQ)
			L = 1 - C
			Y_kp1k = L * M * L + C * invQ * C 
			y_kp1k = L * invF * y_kk 

			# innovate 
			mat_I = 0.0 # np.shape(H_kp1[0]) 
			vec_I = np.zeros((self.param.nq))
			for agent_j in self.env.agents: 
				if adjacency_matrix[agent_i.i,agent_j.i] > 0:
					measurement = z_kp1[:,agent_j.i] 
					measurement_model = H_kp1[:,agent_j.i] 
					mat_I += measurement_model * invR * measurement_model 
					vec_I += measurement_model * invR * measurement

			# get learning rate
			S = H_kp1[:,agent_i.i]*Y_kp1k*H_kp1[:,agent_i.i]+R
			invS = 1/S
			K_kp1[agent_i.i] = Y_kp1k*H_kp1[:,agent_i.i]*invS 

			# invert information transformation
			Y_kp1kp1 = Y_kp1k + mat_I
			y_kp1kp1 = y_kp1k + vec_I
			p_kp1[:,agent_i.i] = 1 / Y_kp1kp1
			r_kp1[:,agent_i.i] = p_kp1[:,agent_i.i] * y_kp1kp1 

		return r_kp1,p_kp1,K_kp1

	
	def make_adjacency_matrix(self):

		A = np.zeros((self.param.ni,self.param.ni))
		for agent_i in self.env.agents:
			p_i = np.array([agent_i.x,agent_i.y])
			for agent_j in self.env.agents:
				p_j = np.array([agent_j.x,agent_j.y])
				dist = np.linalg.norm(p_i-p_j)
				if dist < self.param.r_comm:
					A[agent_i.i,agent_j.i] = 1

		# A = np.zeros((self.param.ni,self.param.ni))
		# for agent_i in self.env.agents:
		# 	p_i = np.array([agent_i.x,agent_i.y])
		# 	for agent_j in self.env.agents:
		# 		if not agent_i is agent_j:
		# 			p_j = np.array([agent_j.x,agent_j.y])
		# 			dist = np.linalg.norm(p_i-p_j)
		# 			if dist < self.param.r_comm:
		# 				A[agent_i.i,agent_j.i] = 1

		# normalize 
		# for agent_i in self.env.agents:
		# 	A[agent_i.i,:] /= sum(A[agent_i.i,:])

		return A

	def get_measurements(self):
		# output
		# 	- z_kp1 : reward measurement for each agent, numpy in nq x ni 
		# 	- H_kp1 : measurement model for each agent, numpy in 1 x ni 

		z_kp1 = np.zeros((self.param.nq, self.param.ni))
		H_kp1 = np.zeros((1,self.param.nq))

		for agent in self.env.agents:
			
			measurement = np.zeros(self.param.nq)

			if agent.update:
				agent.update = False

				measurement = np.zeros((self.param.nq))
				px = agent.service.x_p
				py = agent.service.y_p
				time_of_request = agent.service.time
				time_diff = self.param.sim_times[self.env.timestep] - time_of_request

				for s in range(self.param.env_ncell):
					for a in range(self.param.env_naction):
						q_idx = self.env.sa_to_q_idx(s,a)
						reward_instance = self.env.reward_instance(s,a,px,py,time_diff)
						measurement[q_idx] = reward_instance

				H_kp1[:,agent.i] = 1.0 
			z_kp1[:,agent.i] = measurement 

		return z_kp1,H_kp1  


	# def dtd(self,free_agents):
	# 	# distributed temporal difference

	# 	# get stuff 
	# 	r_k = np.zeros((self.param.nq,self.param.ni))
	# 	q_k = np.zeros((self.param.nq,self.param.ni))
	# 	p_k = np.zeros((self.param.nq,self.param.ni))
	# 	Pq_k = np.zeros((self.param.nq,self.param.nq,self.param.ni))
	# 	q_kp1 = np.zeros((self.param.nq,self.param.ni))
	# 	for agent in self.env.agents:
	# 		r_k[:,agent.i] = agent.r 
	# 		q_k[:,agent.i] = agent.q 
	# 		p_k[:,agent.i] = agent.p 
	# 		Pq_k[:,:,agent.i] = self.env.get_MDP_Pq(agent.q)

	# 	# measurements 
	# 	z_kp1,H_kp1 = self.get_measurements()

	# 	# dkif on reward 
	# 	print('dkif...')
	# 	r_kp1,p_kp1 = self.dkif(r_k,p_k,z_kp1,H_kp1)
	
	# 	# temporal difference
	# 	alpha = 0.05 # temp 
	# 	for agent in self.env.agents:
	# 		td_error = r_kp1[:,agent.i]+self.param.mdp_gamma*np.dot(Pq_k[:,:,agent.i],q_k[:,agent.i])-q_k[:,agent.i]
	# 		q_kp1[:,agent.i] = q_k[:,agent.i] + alpha*td_error
		
	# 	# update agents
	# 	for agent in self.env.agents:
	# 		agent.r = r_kp1[:,agent.i]
	# 		agent.p = p_kp1[:,agent.i]
	# 		agent.q = q_kp1[:,agent.i]

	# 	# task assignment 
	# 	print('ta...')
	# 	cell_assignments = self.ta(self.env,free_agents)

	# 	# assignment 
	# 	move_assignments = self.cell_to_move_assignments(cell_assignments)

	# 	return move_assignments 

	# def ctd(self,free_agents):
	# 	# centralized temporal difference

	# 	# get stuff 
	# 	r_k = self.env.agents[0].r 
	# 	q_k = self.env.agents[0].q
	# 	p_k = self.env.agents[0].p
	# 	Pq_k = self.env.get_MDP_Pq(self.env.agents[0].q)

	# 	# measurements 
	# 	z_kp1,H_kp1 = self.get_measurements()

	# 	# ckif on reward 
	# 	print('ckif...')
	# 	r_kp1,p_kp1 = self.ckif(r_k,p_k,z_kp1,H_kp1)

	# 	# temporal difference
	# 	alpha = 0.05 # temp 
	# 	q_kp1 = q_k + alpha*(r_kp1+self.param.mdp_gamma*np.dot(Pq_k,q_k)-q_k) 
		
	# 	# update agents
	# 	for agent in self.env.agents:
	# 		agent.r = r_kp1 
	# 		agent.p = p_kp1 
	# 		agent.q = q_kp1 

	# 	# task assignment 
	# 	print('ta...')
	# 	cell_assignments = self.ta(self.env,free_agents)

	# 	# assignment 
	# 	move_assignments = self.cell_to_move_assignments(cell_assignments)

	# 	return move_assignments 


	# def ckif(self):

	# 	q_t = self.env.agents[0].q
	# 	p_k = np.zeros((self.param.nq))
		
	# 	# measurements are \hat{R}^i_t
	# 	measurements = self.get_measurements()

	# 	# get matrices 
	# 	F = 1.0
	# 	Q = self.param.process_noise
	# 	R = self.param.measurement_noise
	# 	invF = 1.0 
	# 	invQ = 1.0/self.param.process_noise
	# 	invR = 1.0/self.param.measurement_noise

	# 	# innovate and measurement 
	# 	mat_I = 0.0 
	# 	vec_I = np.zeros((self.param.nq))
	# 	H = np.zeros((self.param.ni),dtype=np.float32)
	# 	for agent,measurement in measurements:
	# 		if np.count_nonzero(measurement) > 0:
	# 			H[agent.i] = 1.0 
	# 		mat_I += H[agent.i] * invR * H[agent.i] 
	# 		vec_I += H[agent.i] * invR * measurement

	# 	update_sum = np.zeros((self.param.nq))
	# 	normalize_sum = 0
	# 	for agent, measurement in measurements:

	# 		# information transformation
	# 		Y_km1km1 = 1/agent.p
	# 		y_km1km1 = Y_km1km1 * agent.q

	# 		# predict 
	# 		M = invF * Y_km1km1 * invF
	# 		C = M / (M + invQ)
	# 		L = 1 - C
	# 		Y_kkm1 = L * M * L + C * invQ * C 
	# 		y_kkm1 = L * invF * y_km1km1 

	# 		# innovate step outside loop 

	# 		# invert information transformation
	# 		Y_kk = Y_kkm1 + mat_I
	# 		y_kk = y_kkm1 + vec_I
	# 		p_k = 1 / Y_kk

	# 		# learning rate
	# 		S = H[agent.i] * Y_kkm1 * H[agent.i] + R
	# 		invS = 1/S 
	# 		alpha = Y_kkm1 * H[agent.i] * invS 

	# 		# td 
	# 		Pq = self.env.get_MDP_Pq(agent.q)
	# 		td_error = measurement + np.dot(self.param.mdp_gamma*Pq - np.eye(self.param.nq), agent.q)

	# 		update_sum += alpha*td_error
	# 		normalize_sum += alpha

	# 	if normalize_sum > 0:
	# 		update_sum /= normalize_sum
		
	# 	q_tp1 = q_t + update_sum 

	# 	for agent in self.env.agents:
	# 		agent.q = q_tp1
	# 		agent.p = p_k

	# def dkif_ms(self):
	# 	# distributed kalman information filter (measurement sharing variant)
		
	# 	measurements = self.get_measurements()
	# 	adjacency_matrix = self.make_adjacency_matrix()

	# 	# init 
	# 	q_kp1 = np.zeros((self.param.nq,self.param.ni))
	# 	p_kp1 = np.zeros((self.param.ni))

	# 	# get matrices 
	# 	F = 1.0
	# 	Q = self.param.process_noise
	# 	R = self.param.measurement_noise
	# 	invF = 1.0
	# 	invQ = 1.0/self.param.process_noise
	# 	invR = 1.0/self.param.measurement_noise

	# 	# get measurement matrices
	# 	H = np.zeros((self.param.ni),dtype=np.float32)
	# 	for agent,measurement in measurements:
	# 		if np.count_nonzero(measurement) > 0:
	# 			H[agent.i] = 1

	# 	# get Pqs
	# 	Pqs = np.zeros((self.param.nq,self.param.nq,self.param.ni))
	# 	for agent,_ in measurements:
	# 		Pqs[:,:,agent.i] = self.env.get_MDP_Pq(agent.q)

	# 	# get learning rates and covariance 
	# 	alpha = np.zeros((self.param.ni))
	# 	for agent,measurement in measurements:
	# 		# information transform (IT)
	# 		Y_km1km1 = 1/agent.p
	# 		y_km1km1 = Y_km1km1*agent.q

	# 		# predict 
	# 		M = invF * Y_km1km1 * invF
	# 		C = M/(M+invQ)
	# 		L = 1 - C 
	# 		Y_kkm1 = L*M*L + C*invQ*C 
	# 		y_kkm1 = L*invF*y_km1km1 

	# 		# innovate 
	# 		mat_I = H[agent.i] * invR
	# 		vec_I = H[agent.i] * invR * measurement
			
	# 		# learning rate 
	# 		S = H[agent.i] * Y_kkm1 * H[agent.i] + R
	# 		invS = 1/S
	# 		alpha[agent.i] = Y_kkm1 * H[agent.i] * invS 

	# 		# invert IT 
	# 		Y_kk = Y_kkm1 + mat_I
	# 		y_kk = y_kkm1 + vec_I
	# 		P_k = 1 / Y_kk
	# 		p_kp1[agent.i] = P_k 

	# 	# reward update 
	# 	reward_term = np.zeros((self.param.nq, self.param.ni))
	# 	# normalize_sum = 0.
	# 	for agent, measurement in measurements:
	# 		if H[agent.i] > 0:
	# 			reward_term[:,agent.i] = alpha[agent.i] * (measurement+np.dot(self.param.mdp_gamma*Pqs[:,:,agent.i]-np.eye(self.param.nq), agent.q))
	# 	# 		normalize_sum += alpha[agent.i]
	# 	# if normalize_sum > 0:
	# 	# 	consensus_term /= normalize_sum


	# 	# consensus update 
	# 	consensus_term = np.zeros((self.param.nq, self.param.ni))
	# 	normalize_sum = 0.
	# 	for agent_i,_ in measurements:
	# 		for agent_j, measurement_j in measurements:
	# 			if adjacency_matrix[agent_i.i,agent_j.i] > 0: 
	# 				# consensus_term[:,agent_i.i] += alpha[agent_j.i]*np.dot(np.eye(self.param.nq)-self.param.mdp_gamma*Pqs[:,:,agent_j.i], agent_j.q-agent_i.q)
	# 				consensus_term[:,agent_i.i] += alpha[agent_j.i]*(measurement_j+np.dot(self.param.mdp_gamma*Pqs[:,:,agent_j.i]-np.eye(self.param.nq), agent_i.q))
	# 				normalize_sum += alpha[agent_j.i]
		
	# 	if normalize_sum > 0:
	# 		consensus_term /= normalize_sum

	# 	# explicit update 
	# 	for agent in self.env.agents:
	# 		q_kp1[:,agent.i] = agent.q + reward_term[:,agent.i] + consensus_term[:,agent.i]

	# 	# assign 
	# 	for agent in self.env.agents:
	# 		agent.q = q_kp1[:,agent.i]
	# 		agent.p = p_kp1[agent.i]

	# def dkif_ms(self):
	# 	# distributed kalman information filter (measurement sharing variant)
		
	# 	measurements = self.get_measurements()
	# 	adjacency_matrix = self.make_adjacency_matrix()

	# 	# init 
	# 	q_kp1 = np.zeros((self.param.nq,self.param.ni))
	# 	p_kp1 = np.zeros((self.param.ni))

	# 	# get matrices 
	# 	F = 1.0
	# 	Q = self.param.process_noise
	# 	R = self.param.measurement_noise
	# 	invF = 1.0
	# 	invQ = 1.0/self.param.process_noise
	# 	invR = 1.0/self.param.measurement_noise
	# 	H = np.zeros((self.param.ni),dtype=np.float32)
	# 	for agent,measurement in measurements:
	# 		if np.count_nonzero(measurement) > 0:
	# 			H[agent.i] = 1

	# 	for agent_i,measurement_i in measurements:

	# 		# information transform (IT)
	# 		Y_km1km1 = 1/agent_i.p
	# 		y_km1km1 = Y_km1km1*agent_i.q

	# 		# predict 
	# 		M = invF * Y_km1km1 * invF
	# 		C = M/(M+invQ)
	# 		L = 1 - C 
	# 		Y_kkm1 = L*M*L + C*invQ*C 
	# 		y_kkm1 = L*invF*y_km1km1 

	# 		# innovate 
	# 		mat_I = 0.0 
	# 		vec_I = np.zeros((self.param.nq))
	# 		for agent_j, measurement_j in measurements:
	# 			if adjacency_matrix[agent_i.i,agent_j.i] > 0:
	# 				mat_I += H[agent_j.i] * invR
	# 				vec_I += H[agent_j.i] * invR * measurement_j

	# 		# invert IT 
	# 		Y_kk = Y_kkm1 + mat_I
	# 		y_kk = y_kkm1 + vec_I
	# 		P_k = 1 / Y_kk
	# 		q_kp1[:,agent_i.i] = P_k * y_kk
	# 		p_kp1[agent_i.i] = P_k # * np.ones((p_kp1.shape[0]))

	# 	# assign 
	# 	for agent in self.env.agents:
	# 		agent.q = q_kp1[:,agent.i]
	# 		agent.p = p_kp1[agent.i]


	# def dkif_ss(self):
	# 	# distributed kalman information filter (state sharing variant)
		
	# 	measurements = self.get_measurements()
	# 	adjacency_matrix = self.make_adjacency_matrix()
	# 	q_kp1 = np.zeros((self.param.nq,self.param.ni))
	# 	p_kp1 = np.zeros((self.param.nq,self.param.ni))

	# 	# get matrices
	# 	F = np.eye(self.param.nq)
	# 	Q = self.param.process_noise*np.eye(self.param.nq)
	# 	R = self.param.measurement_noise*np.eye(self.param.nq)
	# 	invF = np.eye(self.param.nq)
	# 	invQ = 1.0/self.param.process_noise*np.eye(self.param.nq)
	# 	invR = 1.0/self.param.measurement_noise*np.eye(self.param.nq)

	# 	for agent_i,measurement_i in measurements:

	# 		H = np.zeros((self.param.nq,self.param.nq))
	# 		if np.count_nonzero(measurement_i) > 0:
	# 			H = np.eye(self.param.nq)

	# 		# information transformation
	# 		Y_km1km1 = np.diag(1/agent_i.p) 
	# 		y_km1km1 = np.dot(Y_km1km1,agent_i.q)

	# 		# predict
	# 		M = np.dot(invF.T, np.dot(Y_km1km1,invF))
	# 		C = np.dot(M, np.linalg.pinv(M + invQ))
	# 		L = np.eye(self.param.nq) - C 
	# 		Y_kkm1 = np.dot(L,np.dot(M,L.T)) + np.dot(C,np.dot(invQ,C.T))
	# 		y_kkm1 = np.dot(L,np.dot(invF.T,y_km1km1))

	# 		# innovate
	# 		mat_I = np.dot(H.T, np.dot(invR, H))
	# 		vec_I = np.dot(H.T, np.dot(invR, measurement_i))

	# 		# invert information transformation
	# 		Y_kk = Y_kkm1 + mat_I
	# 		y_kk = y_kkm1 + vec_I
	# 		P_k = np.linalg.pinv(Y_kk)
	# 		q_kp1[:,agent_i.i] = np.dot(P_k,y_kk)
	# 		p_kp1[:,agent_i.i] = np.diag(P_k) 

	# 	for agent in self.env.agents:
	# 		agent.q = q_kp1[:,agent.i]
	# 		agent.p = p_kp1[:,agent.i]

	# 	for agent_i in self.env.agents:
	# 		consensus_update = np.zeros((self.param.nq))
	# 		for agent_j,measurement_j in measurements:
	# 			if adjacency_matrix[agent_i.i,agent_j.i] > 0 and np.count_nonzero(measurement_j) > 0:
	# 				consensus_update += agent_j.q-agent_i.q
	# 		agent_i.q += consensus_update