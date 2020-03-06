

import numpy as np 


class Param:
	def __init__(self):

		# flags 
		self.env_render_on = False
		self.plot_sim_over_time = False
		self.make_dataset_on = True
		self.plot_arrows_on = True 

		self.n_trials = 5
		self.results_dir = "../results"

		self.controller_names = [
			'dtd',
			'ctd',
			'bellman', 
			'rhc',
			# 'empty',
			# 'random',
			]

		self.ta = [
			# 'clp',
			'blll',
			# 'da',
		]

		# plotting
		self.plot_fn = 'plots.pdf'
		self.mode_names = ['dispatch','service']
		self.plot_agent_mode_color = ['blue','orange'] 
		self.plot_customer_color = 'orange'
		self.plot_markers = ['s','p','P','*','+']
		self.plot_colors = ['b','g','r','c','m']

		self.state_keys = [
			'gmm_distribution',
			'customers_location',
			'agents_value_fnc_distribution',
			'agents_location',
			'free_agents_distribution',
			'agents_operation',
			'agents_ave_vec_action_distribution'
		]

		# environment parameters
		self.env_name = 'gridworld'
		if self.env_name is 'gridworld':
			
			# state space
			self.env_xlim = [0,4.0]
			self.env_ylim = [0,2.0] #,1]
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
			self.env_lengthscale = ((self.env_xlim[1] - self.env_xlim[0])**2 + (self.env_ylim[1] - self.env_ylim[0])**2)**(1/2)

			# simulation parameters
			self.sim_tf = 100
			self.sim_dt = 0.25
			self.sim_times = np.arange(0,self.sim_tf+self.sim_dt,self.sim_dt)
			self.sim_nt = len(self.sim_times)

			# fleet parameters
			self.ni = 100
			self.r_comm = 3*self.env_dx
			self.r_sense = self.env_lengthscale # 2*self.r_comm
			# self.lambda_a = 1.0 # (cost of customer waiting time)/(cost of agent movement)

			# customer model/dataset 
			self.cm_linear_move = False
			if self.cm_linear_move:
				self.cm_ng = 2
				self.cm_sigma = 0.05 
				self.cm_speed = (self.env_xlim[1] - self.env_xlim[0] - self.env_dx) / self.sim_tf
				self.cm_nsample_cm = 100
				self.n_customers_per_time = 3 # int(0.5*self.ni)
				self.n_training_data = 100
			else:
				self.cm_ng = 2
				self.cm_sigma = 0.05 # ~1/4 env dx -> 2 sigma rule within a dx
				self.cm_speed = 0.1 # 1/10 taxi speed?
				self.cm_nsample_cm = 100
				self.n_customers_per_time = 3 # int(0.5*self.ni)
				self.n_training_data = 100

			
			# choose taxi speed based on swarm parameter = ni * taxi_speed / (n_customers_per_time * env_lengthscale)
			desired_swarm_param = 1.0 
			self.taxi_speed = desired_swarm_param * self.n_customers_per_time * self.env_lengthscale / self.ni 

			# estimation parameters
			self.initial_covariance = 0.0001 
			self.process_noise = self.cm_speed
			self.measurement_noise = self.cm_sigma 

			# mdp parameters
			self.lambda_r = 0.1 #0.8
			self.mdp_gamma = 0.9
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

	def to_dict(self):
		return self.__dict__
