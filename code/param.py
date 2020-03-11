

import numpy as np 


class Param:
	def __init__(self):


		# self.env_name = 'gridworld'
		self.env_name = 'citymap' 

		# flags 
		self.env_render_on = False
		self.plot_sim_over_time = True
		self.make_dataset_on = False
		self.plot_arrows_on = True 

		self.n_trials = 1
		self.results_dir = "../results"

		self.controller_names = [
			['dtd','blll'],
			# ['dtd','clp'],
			# ['ctd','clp'],
			# ['bellman','clp'],
			# ['rhc','clp'],
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

		# action space
		self.env_naction = 5 


		# environment parameters
		if self.env_name is 'gridworld':
			
			# parameters
			
			# sim 
			self.sim_tf = 10
			self.sim_dt = 0.25

			# fleet 
			self.ni = 20
			
			self.desired_env_ncell = 20 # self.env_nx*self.env_ny
			self.desired_swarm_density = 5.0 # agents/m^2
			self.desired_swarm_param = 1.0 
			self.desired_agents_per_cell = 1.0 

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
			self.mdp_gamma = 0.99
			self.mdp_max_iter = 1000
			self.mdp_eps = 1e-4

			# task assignment 
			self.beta = 15 # 150.
			self.ta_converged = 20
			self.ta_tau = 0.0001
			self.ta_tau_decay = 0.1
			self.ta_tau_decay_threshold = 1000
			self.blll_iter_lim_per_agent = 50

			# plot 
			self.plot_r_agent = 0.05
			self.plot_r_customer = 0.05

		elif self.env_name is 'citymap':

			# determine from data 
			self.taxi_speed = 0.1 # temp 
			self.process_noise = 1.0
			self.measurement_noise = 1.0 

			# 
			self.city = 'chicago'
			self.shp_path = '../data/maps/{}.shp'.format(self.city)


			# sim 
			self.sim_tf = 10
			self.sim_dt = 0.25

			# fleet 
			self.ni = 20
			
			self.desired_env_ncell = 50 # self.env_nx*self.env_ny
			self.desired_swarm_density = 5.0 # agents/m^2
			self.desired_swarm_param = 1.0 
			self.desired_agents_per_cell = 1.0 

			# estimation
			self.initial_covariance = 0.0001 

			# mdp 
			self.lambda_r = 0.1 #0.8
			self.mdp_gamma = 0.99
			self.mdp_max_iter = 1000
			self.mdp_eps = 1e-4

			# task assignment 
			self.beta = 15 # 150.
			self.ta_converged = 20
			self.ta_tau = 0.0001
			self.ta_tau_decay = 0.1
			self.ta_tau_decay_threshold = 1000
			self.blll_iter_lim_per_agent = 50

			# plot 
			self.plot_r_agent = 0.05
			self.plot_r_customer = 0.05			

			# timestamp for data gen
			self.train_start_year = 2015
			self.train_start_month = 1
			self.train_start_day = 9
			self.train_start_hour = 23
			self.train_start_minute = 45
			self.train_start_second = 0
			self.train_start_microsecond = 0

			self.train_end_year = 2015
			self.train_end_month = 1
			self.train_end_day = 10
			self.train_end_hour = 0
			self.train_end_minute = 0
			self.train_end_second = 0
			self.train_end_microsecond = 0

			self.test_start_year = 2015
			self.test_start_month = 1
			self.test_start_day = 10
			self.test_start_hour = 0
			self.test_start_minute = 0
			self.test_start_second = 0
			self.test_start_microsecond = 0

			self.test_end_year = 2015
			self.test_end_month = 1
			self.test_end_day = 10
			self.test_end_hour = 0
			self.test_end_minute = 15
			self.test_end_second = 0
			self.test_end_microsecond = 0

	def to_dict(self):
		return self.__dict__

	def update(self):
		if self.env_name is 'gridworld':
			# functions 
			# state space
			
			l = (self.ni/self.desired_swarm_density)**(1/2)
			
			if False:
				dx = l / self.desired_env_ncell**(1/2)
			else:
				dx = (self.desired_agents_per_cell / self.desired_swarm_density)**0.5
			
			l = dx * int(l/dx)

			self.env_xlim = [0,l+dx]
			self.env_ylim = [0,l+dx] #,1]

			# customer model
			self.n_customers_per_time = max(int(0.2*self.ni),1)
			if self.cm_linear_move:
				self.cm_speed = (self.env_xlim[1] - self.env_xlim[0] - self.env_dx) / self.sim_tf
			self.taxi_speed = self.desired_swarm_param * self.n_customers_per_time * self.env_lengthscale / self.ni 
			# estimation
			self.process_noise = self.cm_speed
			self.measurement_noise = self.cm_sigma 

		elif self.env_name is 'citymap':
			self.env_lengthscale = ((self.env_xlim[1]-self.env_xlim[0])**2+(self.env_ylim[1]-self.env_ylim[0])**2)**(1/2)
			dx = self.env_lengthscale / self.desired_env_ncell**(1/2)
			self.env_xlim[1] += dx
			self.env_ylim[1] += dx

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

		# fleet 
		self.r_comm = 3*self.env_dx
		self.r_sense = self.env_lengthscale # 2*self.r_comm
		# self.lambda_a = 1.0 # (cost of customer waiting time)/(cost of agent movement)

		# plotting 
		self.plot_arrow_width = self.plot_r_agent/5
		self.plot_arrow_length = 1.2*self.plot_r_agent
		self.plot_arrow_head_width = self.plot_arrow_width*3
		self.plot_arrow_head_length = self.plot_arrow_head_width




