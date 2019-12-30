

import numpy as np 
from numpy.random import random 

np.random.seed(0)

class ETA:

	def __init__(self,param,env):
		self.param = param
		self.env = env
		if param.env_name is 'gridworld':
			self.__call__ = self.gridworld_eta
		elif param.env_name is 'chicago':
			self.__call__ = self.chicago_eta

	def gridworld_eta(self,i,j,t):

		x_i,y_i = utilities.cell_index_to_coordinate(i)
		x_j,y_j = utilities.cell_index_to_coordinate(j)

		d_ij = np.linalg.norm([x_i-x_j,y_i-y_j])
		w = random()*(self.param.wmax-self.param.wmin)+self.param.wmin
		phi = random()*(self.param.phimax-self.param.phimin)+self.param.phimin
		tau = t/self.param.sim_times[-1]

		return d_ij + np.sin(w*t+phi) + 1 

