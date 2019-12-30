

import numpy as np 

import utilities
from agent import IdleMove, Service, Empty
from task_assignment import centralized_linear_program


class Controller():
	def __init__(self,param,env,dispatch_algorithm):
		self.param = param
		self.env = env

		if dispatch_algorithm in ['empty']:
			self.dispatch = self.zero_move
		elif dispatch_algorithm in ['dtd']:
			self.dispatch = self.dtd
			self.init_value_fnc(env)
		elif dispatch_algorithm in ['ctd']:
			self.dispatch = self.ctd
			self.init_value_fnc(env)
		else:
			exit('fatal error: param.controller_name not recognized')
		

	def dtd(self,agents):
		# temp 
		actions = []
		for agent in agents:
			move_x = 0
			move_y = 0 
			move = IdleMove(move_x,move_y)
			actions.append((agent,move))
		return actions


	def ctd(self,agents):
		# centralized temporal difference learning 
		
		# LP task assignment 
		cell_assignments = centralized_linear_program(self.env,agents)

		# gradient update
		# todo 

		# assignment 
		move_assignments = []
		transition = utilities.get_MDP_P(env)
		for a,c in cell_assignments:
			i = np.where(transition[c,utilities.coordinate_to_cell_index(a.x,a.y),:] == 1)[0][0]
			x,y = utilities.random_position_in_cell(i)
			move_vector = np.array([x-a.x,y-a.y])
			move_vector = move_vector/np.linalg.norm(move_vector)*self.param.average_taxi_speed
			move = IdleMove(move_vector[0],move_vector[1])
			move_assignments.append((a,move))

		return move_assignments


	def zero_move(self,agents):
		move_assignments = []
		for agent in agents:
			move_x = 0
			move_y = 0 
			move = IdleMove(move_x,move_y)
			move_assignments.append((agent,move))
		return move_assignments

	def init_value_fnc(self,env):

		train_dataset = env.dataset[env.dataset[:,0]<0,:]
		v = utilities.solve_MDP(env,train_dataset)

		for agent in env.agents:
			agent.v = v 


	def policy(self,observation):
		
		# input: 
		# - observation is a list of customer requests, passed by reference
		# - customer request: [time_of_request,time_to_complete,x_p,y_p,x_d,y_d], np array
		# output: 
		# - action list
		# - if closest and idle: agent's action is servicing a customer request
		# - elif idle: action chosen using some algorithm (d-td, c-td, RHC)
		# - else: action is continue servicing customer 


		# get idle agents 
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
			for agent in idle_agents: 
				dist = np.linalg.norm([agent.x - service.x_p, agent.y - service.y_p])
				if dist < min_dist and dist < self.param.r_sense:
					min_dist = dist 
					assignment = agent.i 
					serviced = True

			if serviced:
				service_assignments.append((self.env.agents[assignment],service)) 
				observation.remove(service)
				idle_agents.remove(self.env.agents[assignment])
			else:
				print('not serviced: ', service)
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


