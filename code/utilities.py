
import numpy as np 
from numpy.random import random 
from mdptoolbox.mdp import ValueIteration, ValueIterationGS, PolicyIterationModified

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