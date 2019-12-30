
import numpy as np 
from numpy.random import random 
from mdptoolbox.mdp import ValueIteration, PolicyIterationModified

from param import Param 

np.random.seed(0)
param = Param()


def cell_index_to_coordinate(i):
	if param.env_name is 'gridworld':
		x = np.remainder(i,param.env_nx)+0.5
		y = np.floor_divide(i,param.env_nx)+0.5
	return x,y


def coordinate_to_cell_index(x,y):
	if param.env_name is 'gridworld':
		i = np.floor(x) + np.floor(y)*param.env_nx
	return int(i)  


def random_position_in_cell(i):
	if param.env_name is 'gridworld':
		x_cell,y_cell = cell_index_to_coordinate(i)
		x = random() + x_cell - 0.5 
		y = random() + y_cell - 0.5 		
	return x,y


def random_position_in_world():	
	x = random()*param.env_nx
	y = random()*param.env_ny
	return x,y 


def solve_MDP(env,dataset):
	P = get_MDP_P(env) # in AxSxS
	R = get_MDP_R(env,dataset) # in SxA
	mdp = ValueIteration(P, R, env.param.mdp_gamma)
	mdp.run()
	V = np.array(mdp.V)
	V = V/sum(V)
	return V 


def get_MDP_P(env):
	# P in AxSxS 
	P = np.zeros((env.param.env_naction,env.param.env_ncell,env.param.env_ncell))

	for s in range(env.param.env_ncell):

		x,y = cell_index_to_coordinate(s)
		x -= 0.5
		y -= 0.5
		
		# 'empty' action  
		P[0,s,s] = 1.

		# 'right' action
		if not x == env.param.env_x[-1]:
			P[1,s,s+1] = 1.
		else:
			P[1,s,s] = 1

		# 'top' action
		if not y == env.param.env_y[-1]:
			next_s = coordinate_to_cell_index(x,y+1)
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
			next_s = coordinate_to_cell_index(x,y-1)
			P[4,s,next_s] = 1. 
		else:
			P[4,s,s] = 1.

		# print('s: ',s)
		# print('x: ',x)
		# print('y: ',y)
		# print('P[:,s,:]:', P[:,s,:])
	# exit()

	return P  


def get_MDP_R(env,dataset):
	# R in SxA
	R = np.empty((env.param.env_ncell,env.param.env_naction))
	count = np.empty(env.param.env_ncell)

	for data in dataset:
		
		tor = data[0]
		ttc = data[1]
		px = data[2]
		py = data[3]
		dx = data[4]
		dy = data[5]

		for s in range(env.param.env_ncell):

			sx,sy = cell_index_to_coordinate(s)
			cwt = env.eta(px,py,sx,sy,tor)
			R[s,:] += cwt*env.param.mdp_lambda_r**(np.abs(tor))
			count[s] += 1 
	
	R /= len(dataset)
	return R  

