

import numpy as np 


class Param:
	def __init__(self):

		# flags 
		self.env_render_on = False
		self.make_dataset_on = True
		self.plot_arrows_on = True 
		self.plot_distribution_error_on = False
		self.plot_value_fnc_on = False

		self.controllers = [
			'dtd',
			'ctd',
			'bellman', 
			# 'rhc',
			# 'empty',
			# 'random',
			]

		# plotting
		self.plot_fn = 'plots.pdf'
		# modes = idle, pickup, dropoff, dispatch
		self.plot_agent_mode_color = ['blue','green','orange','blue'] 
		self.plot_customer_color = ['blue','green','orange','blue']
		
		# environment parameters
		self.env_name = 'gridworld'
		if self.env_name is 'gridworld':
			
			# state space
			self.env_xlim = [0,2]
			self.env_ylim = [0,0.5] #,1]
			self.env_dx = 0.25 # 0.5 # length/cell
			self.env_dy = self.env_dx
			self.env_x = np.arange(self.env_xlim[0],self.env_xlim[1],self.env_dx)
			self.env_y = np.arange(self.env_ylim[0],self.env_ylim[1],self.env_dy)
			self.env_nx = len(self.env_x)
			self.env_ny = len(self.env_y)
			self.env_ncell = self.env_nx*self.env_ny
			self.env_naction = 5
			self.nv = self.env_ncell 
			self.nq = self.env_ncell*self.env_naction

			# simulation parameters
			self.sim_tf = 20
			self.sim_dt = 0.5
			self.sim_times = np.arange(0,self.sim_tf,self.sim_dt)
			self.sim_nt = len(self.sim_times)

			# fleet parameters
			self.ni = 20
			self.r_comm = 3*self.env_dx
			self.r_sense = np.Inf # 2*self.r_comm
			self.taxi_speed = 1. # dist/time
			self.lambda_a = 1. # (cost of customer waiting time)/(cost of agent movement)

			# customer model/dataset 
			self.cm_linear_move = True
			self.cm_ng = 1
			self.cm_sigma = 0.00001 # self.env_dx # 0.1 #self.env_dx*0.001
			self.cm_speed = self.env_x[-1] / self.sim_tf # 0.25 # dist/time
			self.cm_nsample_cm = 100
			self.n_customers_per_time = 5 #int(0.5*self.ni)
			self.n_training_data = 100

			# estimation parameters
			self.initial_covariance = 0.0001 
			self.process_noise = self.cm_speed
			self.measurement_noise = self.cm_sigma 

			# mdp parameters
			self.lambda_r = 1.0 #0.8
			self.mdp_gamma = 0.99
			self.mdp_max_iter = 1000
			self.mdp_eps = 1e-4

			# normalizing value fnc constant
			self.beta = 150 # 15.
			
			# task assignment parameters
			self.ta_converged = 20
			self.ta_tau = 0.0001
			self.ta_tau_decay = 0.1
			self.ta_tau_decay_threshold = 1000

			# plotting parameters 
			self.plot_r_agent = 0.05
			self.plot_r_customer = 0.05
			self.plot_arrow_width = self.plot_r_agent/5
			self.plot_arrow_length = 1.2*self.plot_r_agent
			self.plot_arrow_head_width = self.plot_arrow_width*3
			self.plot_arrow_head_length = self.plot_arrow_head_width
