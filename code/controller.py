
# standard package
import numpy as np 

# my packages
from helper_classes import Dispatch, Service, Empty
from task_assignment import binary_log_learning
from baseline import centralized_linear_program


class Controller():
	def __init__(self,param,env,dispatch):
		self.param = param
		self.env = env

		if dispatch in ['empty']:
			self.dispatch_name = 'empty'
			self.dispatch = self.empty
		elif dispatch in ['random']:
			self.dispatch_name = 'random'
			self.dispatch = self.random
		elif dispatch in ['H-TD^2']:
			self.dispatch_name = 'htd'
			self.dispatch = self.htd
		elif dispatch in ['D-TD']:
			self.dispatch_name = 'dtd'
			self.dispatch = self.dtd
		elif dispatch in ['C-TD']:
			self.dispatch_name = 'ctd'
			self.dispatch = self.ctd
		elif dispatch in ['RHC']:
			self.dispatch_name = 'rhc'
			self.dispatch = self.rhc
		elif dispatch in ['Bellman']:
			self.dispatch_name = 'bellman'
			self.dispatch = self.bellman

		self.ta = binary_log_learning
		self.name = dispatch

	# ------------ simulator -------------

	def policy(self,observation):
		
		# input: 
		# - observation is a list of customer requests, passed by reference (so you can remove customers when serviced)
		# - customer request: [time_of_request,time_to_complete,x_p,y_p,x_d,y_d], np array
		# output: 
		# - action list of dispatch position vectors 

		# action:
		# - if currently servicing customer: continue
		# - elif: closest agent to some 
		# - else: dispatch action chosen using some algorithm (d-td, c-td, RHC, etc)


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


		# dtd 
		print('dkif...')
		r_kp1,p_kp1,K_kp1 = self.dkif(r_k,p_k,z_kp1,H_kp1)

		# get A matrix 
		A_k = self.make_A(K_kp1,H_kp1)

		# 
		for agent_i in self.env.agents:
			r_kp1[:,agent_i.i] = agent_i.r + np.dot(A_k[agent_i.i,:], z_kp1.T - agent_i.r)

		# temporal difference
		alpha = self.param.td_alpha
		for agent in self.env.agents:
			td_error = r_kp1[:,agent.i]+self.param.mdp_gamma*np.dot(Pq_k[:,:,agent.i],q_k[:,agent.i])-q_k[:,agent.i]
			q_kp1[:,agent.i] = q_k[:,agent.i] + alpha*td_error

		# error
		delta_e = self.env.calc_delta_e(A_k)
		delta_d = self.env.calc_delta_d()

		print('delta_e: ', delta_e)
		print('delta_d: ', delta_d)
		if delta_e > delta_d and self.env.timestep > self.env.reset_timestep + self.param.htd_time_window: 
			# bellman
			print('bellman...')
			self.env.reset_timestep = self.env.timestep
			self.env.lambda_min[:,self.env.timestep] = np.ones((self.param.ni))
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

		# 		
		cell_assignments = centralized_linear_program(self.env,agents)

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

		print('cell_to_move_assignments...')
		
		move_assignments = [] 
		# transition = self.env.get_MDP_P(self.env)
		transition = self.env.P
		for agent,cell in cell_assignments:
			i = np.where(transition[cell,self.env.coordinate_to_cell_index(agent.x,agent.y),:] == 1)[0][0]
						
			if True:
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
		# 	- K_kp1 : next gains, numpy in ni 

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
			mat_I = sum(
				np.dot(H_kp1[:,adjacency_matrix[agent_i.i,:] > 0],invR*H_kp1[:,adjacency_matrix[agent_i.i,:] > 0].T)
				)
			vec_I = sum(
				np.dot(H_kp1[:,adjacency_matrix[agent_i.i,:] > 0],invR*z_kp1[:,adjacency_matrix[agent_i.i,:] > 0].T)
				)

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

	
	def make_A(self,K,H):

		A = np.zeros((self.param.ni,self.param.ni))
		for agent_i in self.env.agents:
			p_i = np.array([agent_i.x,agent_i.y])
			for agent_j in self.env.agents:
				p_j = np.array([agent_j.x,agent_j.y])
				dist = np.linalg.norm(p_i-p_j)
				if dist < self.param.r_comm:
					A[agent_i.i,agent_j.i] = K[agent_j.i] * H[:,agent_j.i] 
		# normalize 
		for agent_i in self.env.agents:
			if sum(A[agent_i.i,:]) > 0: 
				A[agent_i.i,:] /= sum(A[agent_i.i,:])
		return A

	def make_adjacency_matrix(self):

		A = np.zeros((self.param.ni,self.param.ni))
		for agent_i in self.env.agents:
			p_i = np.array([agent_i.x,agent_i.y])
			for agent_j in self.env.agents:
				p_j = np.array([agent_j.x,agent_j.y])
				dist = np.linalg.norm(p_i-p_j)
				if dist < self.param.r_comm:
					A[agent_i.i,agent_j.i] = 1
		return A		

	def get_measurements(self):
		# output
		# 	- z_kp1 : reward measurement for each agent, numpy in nq x ni 
		# 	- H_kp1 : measurement model for each agent, numpy in 1 x ni 

		z_kp1 = np.zeros((self.param.nq, self.param.ni))
		H_kp1 = np.zeros((1,self.param.ni))

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