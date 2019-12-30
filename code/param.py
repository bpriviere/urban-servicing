

import numpy as np 
np.random.seed(0)

class Param:
	def __init__(self):

		
		# environment parameters
		self.env_name = 'gridworld'

		if self.env_name is 'gridworld':

			# flags
			self.env_render_on = True 
			
			# state space
			self.env_xlim = [0,2]
			self.env_ylim = [0,2]
			self.env_dx = 1
			self.env_dy = 1 
			self.env_x = np.arange(self.env_xlim[0],self.env_xlim[1],self.env_dx)
			self.env_y = np.arange(self.env_ylim[0],self.env_ylim[1],self.env_dy)
			self.env_nx = len(self.env_x)
			self.env_ny = len(self.env_y)
			self.env_ncell = self.env_nx*self.env_ny
			self.env_naction = 5
			self.nq = self.env_ncell #2 # np.min([self.max_neighbor_cells*self.env_ncell*self.env_ncell, self.env_ncell**3])

			# simulation parameters
			self.sim_nt = 1
			self.sim_dt = 0.1
			self.sim_times = np.arange(0,self.sim_nt,self.sim_dt)

			# customer requests model
			self.wmax = 1*np.pi
			self.wmin = 1/4*np.pi
			self.phimax = 2*np.pi
			self.phimin = 0.

			# mdp parameters
			self.mdp_lambda_r = 0.8
			self.mdp_gamma = 0.8

			# 
			self.average_taxi_speed = 1. # dist/time


		# fleet parameters
		self.ni = 10
		self.r_comm = 0.5
		self.r_sense = np.Inf # 2*self.r_comm

		# data parameters
		self.make_dataset_on = True
		self.n_customers_per_time = 10 #int(0.5*self.ni)
		self.n_training_data = 10


		# controller parameters: #[ empty, ctd, dtd, ]
		self.controllers = [
			'ctd',
			]

		# plotting
		self.plot_fn = 'plots.pdf'
		self.plot_r_agent = 0.05
		self.plot_r_customer = 0.05
		self.plot_agent_mode_color = ['blue','gray','gray']
		self.plot_customer_color = ['green','gray','gray']
