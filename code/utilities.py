
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
	x = random()*(param.env_x[-1]-param.env_x[0]) + param.env_x[0]
	y = random()*(param.env_y[-1]-param.env_y[0]) + param.env_y[0]
	return x,y 


def environment_barrier(p):
	x = np.clip(p[0],param.env_x[0],param.env_x[-1]+param.env_dx)
	y = np.clip(p[1],param.env_y[0],param.env_y[-1]+param.env_dy)
	return x,y


def softmax(x):
	softmax = np.exp(param.beta*x)/sum(np.exp(param.beta*x))
	return softmax

def solve_MDP(env,dataset):
	print('solve_MDP:')

	print('   get_MDP_P...')
	P = get_MDP_P(env) # in AxSxS
	print('   get_MDP_P complete')
	print('   get_MDP_R...')
	R = get_MDP_R(env,dataset) # in SxA
	print('   get_MDP_R complete')	
	print('   value iteration...')
	mdp = ValueIteration(P, R, env.param.mdp_gamma, env.param.mdp_eps, env.param.mdp_max_iter)
	# mdp = ValueIterationGS(P, R, env.param.mdp_gamma, epsilon=env.param.mdp_eps, max_iter=env.param.mdp_max_iter)
	mdp.setVerbose()
	mdp.run()
	print('   value iteration complete')
	V = np.array(mdp.V)
	
	# print('mdp: ',mdp)
	print('mdp.V: ',mdp.V)
	# print('mdp.policy: ',mdp.policy)
	# exit()
	return V 


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


def get_MDP_R(env,dataset):
	# R in SxA
	R = np.zeros((env.param.env_ncell,env.param.env_naction))
	P = get_MDP_P(env)

	for data in dataset:
		
		tor = data[0]
		px = data[2]
		py = data[3]

		# print('(px,py): ({},{})'.format(px,py))

		for s in range(env.param.env_ncell):
			for a in range(env.param.env_naction):
				next_state = np.where(P[a,s,:] == 1)[0][0]
				sx,sy = cell_index_to_cell_coordinate(next_state)
				sx += env.param.env_dx/2
				sy += env.param.env_dy/2
				cwt = env.eta(px,py,sx,sy,tor)
				action_cost = param.lambda_a*(not a==0)
				cost = cwt + action_cost 
				R[s,a] += -1*cost*env.param.mdp_lambda_r**(np.abs(tor))
				# R[s,a] += -1*cwt

				# print('(s,a,next_state): ({},{},{})'.format(s,a,next_state))
				# print('(sx,sy): ({},{})'.format(sx,sy))
				# print('cwt: ', cwt)

	R /= len(dataset)

	# for s in range(env.param.env_ncell):
	# 	print('s: ',s)
	# 	print('P[:,s,:]: ',P[:,s,:])
	# print('R: ',R)
	# exit()

	return R  

