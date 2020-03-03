
import numpy as np 
from numpy.random import random 
from mdptoolbox.mdp import ValueIteration, ValueIterationGS, PolicyIterationModified

from param import Param 

np.random.seed(0)
param = Param()


# 'cell_index' : element of [0,...,env_ncell]
# 'cell_coordinate' : (x,y) coordinates of bottom left corner of cell 
# 'xy_cell_index' : (i_x,i_y) indices corresponding to elements of env_x, env_y
# 'coordinate' : free (x,y) coordinate, not constrained by being on gridlines


def cell_index_to_cell_coordinate(i):
	if param.env_name is 'gridworld':
		x = param.env_dx*np.remainder(i,param.env_nx)
		y = param.env_dy*np.floor_divide(i,param.env_nx)
	return x,y


def xy_cell_index_to_cell_index(i_x,i_y):
	i = i_y*len(param.env_x) + i_x
	return int(i) 


def cell_index_to_xy_cell_index(i):
	x,y = cell_index_to_cell_coordinate(i)
	i_x,i_y = coordinate_to_xy_cell_index(x,y)
	return i_x,i_y


def coordinate_to_xy_cell_index(x,y):
	i = coordinate_to_cell_index(x,y)
	x,y = cell_index_to_cell_coordinate(i)
	i_x = x/param.env_dx
	i_y = y/param.env_dy
	return int(i_x),int(i_y)


def coordinate_to_cell_index(x,y):
	if param.env_name is 'gridworld':
		i_x = np.where(param.env_x <= x)[0][-1]
		i_y = np.where(param.env_y <= y)[0][-1]
		i = xy_cell_index_to_cell_index(i_x,i_y)
	return int(i)


def random_position_in_cell(i):
	if param.env_name is 'gridworld':
		x,y = cell_index_to_cell_coordinate(i)
		x = param.env_dx*random() + x
		y = param.env_dy*random() + y
	return x,y


def random_position_in_world():	
	x = random()*(param.env_xlim[1] - param.env_xlim[0]) + param.env_xlim[0]
	y = random()*(param.env_ylim[1] - param.env_ylim[0]) + param.env_ylim[0]
	return x,y 


def environment_barrier(p):
	eps = 1e-16
	x = np.clip(p[0],param.env_xlim[0]+eps,param.env_xlim[1]-eps)
	y = np.clip(p[1],param.env_ylim[0]+eps,param.env_ylim[1]-eps)
	return x,y


def softmax(x):
	softmax = np.exp(param.beta*x)/sum(np.exp(param.beta*x))
	return softmax


def value_to_probability(x):
	x = x/abs(sum(x))
	x = softmax(x)
	return x


def get_next_state(env,s,a):
	P = get_MDP_P(env)
	next_state = np.where(P[a,s,:] == 1)[0][0]
	return next_state

def s_sp_to_a(env,s,sp):
	P = get_MDP_P(env)
	next_state = np.where(P[:,s,sp] == 1)[0][0]
	return next_state

def sa_to_q_idx(s,a):
	q_idx = s*param.env_naction + a
	return q_idx 


def get_prev_sa_lst(env,s):
	prev_sa_lst = []
	P = get_MDP_P(env)
	local_s = get_local_states(env,s)
	added_s = []
	for prev_s in local_s:
		for prev_a in range(param.env_naction):
			if P[prev_a,prev_s,s] == 1 and not prev_s in added_s:
				prev_sa_lst.append((prev_s,prev_a))
				added_s.append(prev_s)
	return prev_sa_lst 


def get_local_q_values(env,agent):
	local_q = []
	local_s = []
	s = coordinate_to_cell_index(agent.x,agent.y)
	for a in range(env.param.env_naction):
		next_state = get_next_state(env,s,a)
		if not next_state in local_s:
			local_s.append(next_state)
			local_q.append(agent.q[sa_to_q_idx(s,a)])
	return local_q


def get_local_states(env,s):
	local = []
	for a in range(env.param.env_naction):
		next_state = get_next_state(env,s,a) 
		if not next_state in local:
			local.append(next_state)
	return local 


def reward_instance(env,s,a,px,py):

	# input 
	# 	-env: 
	#	-s: current state
	# 	-a: action
	# 	-px,py: x,y position of customer data
	# output
	# 	-reward: customer waiting time 
	
	sp = get_next_state(env,s,a)
	spx,spy = cell_index_to_cell_coordinate(sp)
	spx += param.env_dx/2
	spy += param.env_dy/2
	sx,sy = cell_index_to_cell_coordinate(s)
	sx += param.env_dx/2
	sy += param.env_dy/2

	time_s_to_sp = env.eta(sx,sy,spx,spy)
	time_sp_to_c = env.eta(spx,spy,px,py)
	reward = -1*(time_s_to_sp + time_sp_to_c)
	
	# action_cost = param.lambda_a*(not a==0)
	# cost = cwt + action_cost 

	return reward

# mdp stuff 
def solve_MDP(env,dataset,curr_time):

	# print('solve_MDP:')

	# print('   get_MDP_P...')
	P = get_MDP_P(env) # in AxSxS
	# print('   get_MDP_P complete')
	# print('   get_MDP_R...')
	R = get_MDP_R(env,dataset,curr_time) # in SxA
	# print('   get_MDP_R complete')	
	# print('   value iteration...')
	mdp = ValueIteration(P,R,env.param.mdp_gamma,env.param.mdp_eps,env.param.mdp_max_iter)
	# mdp = ValueIterationGS(P, R, env.param.mdp_gamma, epsilon=env.param.mdp_eps, max_iter=env.param.mdp_max_iter)
	# mdp.setVerbose()
	mdp.run()
	# print('   value iteration complete')
	# print('mdp.V: ',mdp.V)
	# print('mdp: ',mdp)
	# print('mdp.policy: ',mdp.policy)
	# exit()

	
	V = np.array(mdp.V)
	Q = get_MDP_Q(env,R,V,env.param.mdp_gamma)
		
	return V,Q  


def get_MDP_Q(env,R,V,gamma):

	Q = np.zeros(env.param.env_naction*env.param.env_ncell)
	for s in range(env.param.env_ncell):
		for a in range(env.param.env_naction):
			next_state = get_next_state(env,s,a)
			idx = s*env.param.env_naction + a 
			Q[idx] = R[s,a] + gamma*V[next_state]
	return Q 

def q_value_to_value_fnc(env,q):
	v = np.zeros(param.env_ncell)
	for s in range(env.param.env_ncell):
		idx = s*param.env_naction + np.arange(0,param.env_naction)
		v[s] = max(q[idx])
	return v



def get_MDP_P(env):
	# P in AxSxS 
	P = np.zeros((env.param.env_naction,env.param.env_ncell,env.param.env_ncell))

	for s in range(env.param.env_ncell):

		x,y = cell_index_to_cell_coordinate(s)

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
			next_s = coordinate_to_cell_index(x,y+param.env_dy)
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
			next_s = coordinate_to_cell_index(x,y-param.env_dy)
			P[4,s,next_s] = 1. 
		else:
			P[4,s,s] = 1.

		# print('P[:,s,:]:', P[:,s,:])
	# exit()

	return P  


def get_MDP_R(env,dataset,curr_time):
	# R in SxA
	R = np.zeros((env.param.env_ncell,env.param.env_naction))
	P = get_MDP_P(env)

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
				R[s,a] += reward_instance(env,s,a,px,py)*time_discount
				time_discount_sum[s,a] += time_discount

	if count > 0 :
		R /= count

	R = R/time_discount_sum

	return R  