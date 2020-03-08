

import numpy as np 


class Param:
	def __init__(self):

		# flags 
		self.env_render_on = False
		self.plot_sim_over_time = False
		self.make_dataset_on = True
		self.plot_arrows_on = True 

		self.n_trials = 1
		self.results_dir = "../results"

		self.controller_names = [
			'dtd',
			# 'ctd',
			# 'bellman',
			# 'rhc',
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
			# 'agents_ave_vec_action_distribution'
		]

		# environment parameters
		self.env_name = 'gridworld'
		if self.env_name is 'gridworld':
			
			# parameters
			# state space
			self.desired_env_ncell = 20 # self.env_nx*self.env_ny
			self.env_naction = 5 
			
			# sim 
			self.sim_tf = 2
			self.sim_dt = 0.25

			# fleet 
			self.ni = 20
			self.desired_swarm_density = 10.0 # agents/m^2
			self.desired_swarm_param = 1.0 

			# customer model
			self.cm_linear_move = False
			if self.cm_linear_move:
				self.cm_ng = 2
				self.cm_sigma = 0.05 
				self.cm_nsample_cm = 100
				self.n_training_data = 100
			else:
				self.cm_ng = 2
				self.cm_sigma = 0.05 # ~1/4 env dx -> 2 sigma rule within a dx
				self.cm_speed = 0.1 # 1/10 taxi speed?
				self.cm_nsample_cm = 100
				self.n_training_data = 100

			# estimation
			self.initial_covariance = 0.0001 

			# mdp 
			self.lambda_r = 0.1 #0.8
			self.mdp_gamma = 0.9
			self.mdp_max_iter = 1000
			self.mdp_eps = 1e-4

			# task assignment 
			self.beta = 150 # 15.
			self.ta_converged = 20
			self.ta_tau = 0.0001
			self.ta_tau_decay = 0.1
			self.ta_tau_decay_threshold = 1000

			# plot 
			self.plot_r_agent = 0.05
			self.plot_r_customer = 0.05

	def to_dict(self):
		return self.__dict__

	def update(self):
		if self.env_name is 'gridworld':
			# functions 
			# state space
			
			l = (self.ni/self.desired_swarm_density)**(1/2)
			dx = l / self.ni**(1/2)
			self.env_xlim = [0,l+dx]
			self.env_ylim = [0,l+dx] #,1]

			self.env_dx = dx # 0.5 # length/cell
			self.env_dy = self.env_dx
			self.env_x = np.arange(self.env_xlim[0],self.env_xlim[1],self.env_dx)
			self.env_y = np.arange(self.env_ylim[0],self.env_ylim[1],self.env_dy)
			self.env_nx = len(self.env_x)
			self.env_ny = len(self.env_y)
			self.env_ncell = self.env_nx*self.env_ny
			self.nv = self.env_ncell 
			self.nq = self.env_ncell*self.env_naction
			self.env_lengthscale = ((self.env_xlim[1] - self.env_xlim[0])**2 + (self.env_ylim[1] - self.env_ylim[0])**2)**(1/2)

			# sim 
			self.sim_times = np.arange(0,self.sim_tf+self.sim_dt,self.sim_dt)
			self.sim_nt = len(self.sim_times)

			# customer model
			self.n_customers_per_time = max(int(0.1*self.ni),1)
			if self.cm_linear_move:
				self.cm_speed = (self.env_xlim[1] - self.env_xlim[0] - self.env_dx) / self.sim_tf

			# fleet 
			self.r_comm = 3*self.env_dx
			self.r_sense = self.env_lengthscale # 2*self.r_comm
			# self.lambda_a = 1.0 # (cost of customer waiting time)/(cost of agent movement)
			self.taxi_speed = self.desired_swarm_param * self.n_customers_per_time * self.env_lengthscale / self.ni 

			# estimation
			self.process_noise = self.cm_speed
			self.measurement_noise = self.cm_sigma 

			# plotting 
			self.plot_arrow_width = self.plot_r_agent/5
			self.plot_arrow_length = 1.2*self.plot_r_agent
			self.plot_arrow_head_width = self.plot_arrow_width*3
			self.plot_arrow_head_length = self.plot_arrow_head_width

