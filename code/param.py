

import numpy as np 
from datetime import datetime 

class Param:
	def __init__(self):

		self.verbose = True

		self.load_param = False
		self.load_param_fn = '../results/2020-04-12_testing_param_loader_2/sim_result_1/param.json'

		self.env_name = 'gridworld'
		# self.env_name = 'citymap' 

		self.global_reward_on = True
		self.macro_sim_on = False

		# flags 
		self.env_render_on = False # not implemented
		self.plot_sim_over_time = False
		self.plot_arrows_on = False

		self.n_trials = 1
		self.htd_time_window = 5
		self.htd_minimum_reset = 5
		self.rhc_horizon = 10
		self.delta_d_ratio = 0.025

		self.controller_names = [
			# 'D-TD',
			'H-TD^2',
			# 'C-TD',
			# 'Bellman',
			# 'RHC'
			]


		# environment parameters
		if self.env_name is 'gridworld':
	
			# learning rate
			self.td_alpha = 0.75 #0.5			
			
			# flags
			self.make_dataset_on = True
			self.nc_dilution = 1 
			self.ni_dilution = 1 

			# sim 
			self.sim_t0 = 0 
			self.sim_tf = 10 # 20
			self.sim_dt = 0.1
		
			# parameter tuning with hand picked variables 
			self.swarm_parameters_ver = 2

			if self.swarm_parameters_ver == 2:
				# this one 

				# fleet 
				self.ni = 500 # 20 

				# customer model
				self.cm_taxi_speed_ratio = 0.1
				self.cm_sigma_ratio = 0.25 # std = sigma_ratio * env_dx 
				self.cm_ng = 2 # 1
				self.cm_speed_ratio = 0 * 0.15 # speed = speed_ratio * taxi speed 
				self.n_customers_per_time_ratio = 0.15 # 0.2 
				self.cm_linear_move = False 

				# swarm param 
				self.env_lengthscale = 1.0 # 
				self.desired_env_ncell = 100 # 150 #5*self.ni # 2
				self.desired_aspect_ratio = 2.0 # 2.0 # numx/numy
				self.desired_swarm_param = 1.25 # 1.2 

			elif self.swarm_parameters_ver == 4:
				# recreate smallscale gridworld sim 

				# fleet
				self.ni = 5
				self.taxi_speed = 0.25

				# map
				self.env_dx = 0.25
				self.env_dy = self.env_dx 
				self.env_xlim = [0,0.5]
				self.env_ylim = [0,0.25]
				self.env_lengthscale = self.env_xlim[1] - self.env_xlim[0]

				# customer model
				self.cm_ng = 1 # 1
				self.n_customers_per_time = 1
				self.cm_linear_move = True 
				self.n_customers_per_time_ratio = 0.1 # 0.2 
				self.cm_sigma_ratio = 0.25 # std = sigma_ratio * env_dx 
				self.cm_speed_ratio = 0.15 # speed = speed_ratio * taxi speed 
		
			# estimation
			self.p0 = 0.0001 

			# mdp 
			self.n_training_data = 100
			self.lambda_r = 1.0 #0.9999 #0.2 #0.1
			self.mdp_gamma = 0.9 # 0.8 # 0.8 
			self.mdp_max_iter = 1000
			self.mdp_max_data = 1000
			self.mdp_eps = 1e-4

			# task assignment 
			self.ta_beta = 150 # 150 # 150.
			self.ta_converged = 20
			self.ta_tau = 0.0001
			self.ta_tau_decay = 0.1
			self.ta_tau_decay_threshold = 1000
			self.ta_iter_lim_per_agent = 50

			# plot 
			self.plot_r_agent = 0.05
			self.plot_r_customer = 0.05

		elif self.env_name is 'citymap':

			self.xmin_thresh = -87.8
			self.xmax_thresh = -87.575
			self.ymin_thresh = 41.8 
			self.ymax_thresh = 42.0

			self.make_dataset_on = True

			self.sim_dt = 15*60 

			# learning rate
			self.td_alpha = 0.1 #0.5				

			# determine from data 
			self.taxi_speed = 0.00009 # 22 mph 
			self.process_noise = 0.0001 
			self.measurement_noise = 0.0001 

			# 
			self.city = 'chicago'
			self.shp_path = '../maps/{}.shp'.format(self.city)

			# fleet 
			self.ni_dilution = 1
			self.nc_dilution = 5
			self.ni = int(2000/self.ni_dilution)
			
			self.desired_env_ncell = 300 # self.env_nx*self.env_ny
			self.desired_swarm_density = 5.0 # agents/m^2
			self.desired_swarm_param = 1.0 
			self.desired_agents_per_cell = 1.0 

			# estimation
			self.p0 = 0.01 

			# mdp 
			self.lambda_r = 0.8
			self.mdp_gamma = 0.8
			self.mdp_max_iter = 1000
			self.mdp_eps = 1e-4
			self.mdp_max_data = 1000

			# task assignment 
			self.ta_beta = 1 # 150.
			self.ta_converged = 20
			self.ta_tau = 0.0001
			self.ta_tau_decay = 0.1
			self.ta_tau_decay_threshold = 1000
			self.ta_iter_lim_per_agent = 20

			# plot 
			self.plot_r_agent = 0.05
			self.plot_r_customer = 0.05

			# timestamp for data gen
			self.train_start_year = 2016
			self.train_start_month = 10
			self.train_start_day = 29
			self.train_start_hour = 4
			self.train_start_minute = 0
			self.train_start_second = 0
			self.train_start_microsecond = 0

			self.train_end_year = self.train_start_year
			self.train_end_month = self.train_start_month
			self.train_end_day = self.train_start_day + 1
			self.train_end_hour = 4
			self.train_end_minute = 0
			self.train_end_second = 0
			self.train_end_microsecond = 0

			self.test_start_year = self.train_start_year
			self.test_start_month = self.train_start_month
			self.test_start_day = self.train_start_day + 1
			self.test_start_hour = 4
			self.test_start_minute = 0
			self.test_start_second = 0
			self.test_start_microsecond = 0

			self.test_end_year = self.train_start_year
			self.test_end_month = self.train_start_month
			self.test_end_day = self.train_start_day + 2
			self.test_end_hour = 4
			self.test_end_minute = 0
			self.test_end_second = 0
			self.test_end_microsecond = 0

		# plotting
		self.plot_fn = 'plots.pdf'
		self.mode_names = ['dispatch','service']
		self.plot_agent_mode_color = ['blue','orange'] 
		self.plot_customer_color = 'orange'
		self.plot_markers = ['s','p','P','*','+']
		self.plot_colors = ['b','g','r','c','m']

		self.state_keys = [
			# 'gmm_distribution',
			'customers_location',
			'waiting_customers_location',
			'agents_value_fnc_distribution',
			# 'agents_q_value',
			'agents_location',
			'free_agents_distribution',
			'agents_distribution',
			'agents_operation',
			# 'agents_ave_vec_action_distribution'
		]

		self.plot_keys = [
			'customers_location',
			'waiting_customers_location',
			'agents_value_fnc_distribution',
			'agents_location',
			'reward',
			'free_agents_distribution',
		]

		# action space
		self.env_naction = 5 


	def to_dict(self):
		return self.__dict__

	def load_dict(self):
		
		temp_dict = dict()
		import json
		with open(self.load_param_fn, 'r') as j:
			temp_dict = json.loads(j.read())

		for key,value in temp_dict.items():
			setattr(self,key,value)


	def update(self):
		if self.env_name is 'gridworld':
			# functions 
			# state space
			
			if self.swarm_parameters_ver == 0:
				
				# map 
				self.env_lengthscale = (self.ni/self.desired_swarm_density)**(1/2)
				self.env_dx = (self.desired_agents_per_cell / self.desired_swarm_density)**0.5
				self.env_lengthscale = self.env_dx * np.ceil(self.env_lengthscale/self.env_dx)
				self.env_xlim = [0,self.env_lengthscale]
				self.env_ylim = [0,self.env_lengthscale] #,1]
				self.env_dy = self.env_dx

				# customer model
				self.n_customers_per_time = max((int(0.2*self.ni),1))
				if self.cm_linear_move:
					self.cm_speed = (self.env_xlim[1] - self.env_xlim[0]) / self.sim_tf
				self.taxi_speed = self.desired_swarm_param * self.n_customers_per_time * self.env_lengthscale / self.ni 

			elif self.swarm_parameters_ver == 1:

				# cm 
				self.cm_speed = self.cm_taxi_speed_ratio*self.taxi_speed
				self.n_customers_per_time = max((int(self.n_customers_per_time_ratio*self.ni),1))

				# map 
				self.env_lengthscale = (self.taxi_speed*self.ni)/(self.desired_swarm_param*self.n_customers_per_time)
				self.env_dx = np.sqrt(self.desired_aspect_ratio*self.desired_agents_per_cell*self.env_lengthscale*self.env_lengthscale/self.ni)
				self.env_dy = self.env_dx 
				self.env_xlengthscale = self.env_dx*int(self.env_lengthscale/self.env_dx)
				self.env_ylengthscale = self.env_dy*int((self.env_lengthscale/self.desired_aspect_ratio)/self.env_dy)
				self.env_xlim = [0,self.env_xlengthscale]
				self.env_ylim = [0,self.env_ylengthscale]

			elif self.swarm_parameters_ver == 2:

				# cm
				self.n_customers_per_time = max((int(self.n_customers_per_time_ratio*self.ni),1))
				
				# map 
				self.env_dx = ((self.env_lengthscale*self.env_lengthscale/self.desired_aspect_ratio) / self.desired_env_ncell)**(1/2)
				self.env_dy = self.env_dx
				self.env_xlengthscale = self.env_dx*int(self.env_lengthscale/self.env_dx)
				self.env_ylengthscale = self.env_dy*int((self.env_lengthscale/self.desired_aspect_ratio)/self.env_dy)
				self.env_xlim = [0,self.env_xlengthscale]
				self.env_ylim = [0,self.env_ylengthscale] #,1]
				self.env_dy = self.env_dx

				# map
				self.taxi_speed = self.desired_swarm_param * self.n_customers_per_time * self.env_lengthscale / self.ni 
				
				# cm 
				self.cm_speed = self.cm_taxi_speed_ratio*self.taxi_speed

			elif self.swarm_parameters_ver == 3:
				self.n_customers_per_time = max((int(self.n_customers_per_time_ratio*self.ni),1))
				self.taxi_speed = self.desired_swarm_param * self.n_customers_per_time * self.env_lengthscale / self.ni 
				self.cm_speed = self.cm_taxi_speed_ratio*self.taxi_speed

			# some cm stuff
			if self.cm_linear_move:
				self.cm_ng = 1
				self.cm_sigma = 0.0005
				self.cm_speed = 0.1 * self.taxi_speed
				self.cm_nsample_cm = 100
			else:
				self.cm_sigma = self.cm_sigma_ratio*self.env_dx
				self.cm_speed = self.cm_speed_ratio*self.taxi_speed
				self.cm_nsample_cm = 100

			# estimation
			self.process_noise = self.cm_speed + 1e-9
			self.measurement_noise = self.cm_sigma 

		elif self.env_name is 'citymap':
			
			# lengthscale stuff
			self.env_lengthscale = ((self.env_xlim[1]-self.env_xlim[0])**2+(self.env_ylim[1]-self.env_ylim[0])**2)**(1/2)
			self.env_dx = self.env_lengthscale / self.desired_env_ncell**(1/2)
			self.env_dy = self.env_dx

			# datetime stuff 
			train_end = datetime(self.train_end_year, 
				self.train_end_month, 
				self.train_end_day, 
				self.train_end_hour, 
				self.train_end_minute, 
				self.train_end_second, 
				self.train_end_microsecond) 
			test_end = datetime(self.test_end_year, 
				self.test_end_month, 
				self.test_end_day, 
				self.test_end_hour, 
				self.test_end_minute, 
				self.test_end_second, 
				self.test_end_microsecond) 

			# sim 
			self.sim_t0 = train_end.timestamp() # in sec 
			self.sim_tf = test_end.timestamp()


		# common for all set of parameters
		# env_x,env_y are the bottom left hand corner of each cell in the map 
		self.env_x = np.arange(self.env_xlim[0],self.env_xlim[1],self.env_dx)
		self.env_y = np.arange(self.env_ylim[0],self.env_ylim[1],self.env_dy)
		self.env_nx = len(self.env_x)
		self.env_ny = len(self.env_y)
		self.env_ncell = self.env_nx*self.env_ny
		self.nq = self.env_ncell*self.env_naction
		# self.env_lengthscale = ((self.env_xlim[1] - self.env_xlim[0])**2 + (self.env_ylim[1] - self.env_ylim[0])**2)**(1/2)
		# fleet 
		self.r_comm = 3*self.env_dx
		self.r_sense = self.env_lengthscale # 2*self.r_comm
		# times  
		self.sim_times = np.arange(self.sim_t0,self.sim_tf+self.sim_dt,self.sim_dt)
		self.sim_nt = len(self.sim_times)
		# plotting 
		self.plot_arrow_width = self.plot_r_agent/5
		self.plot_arrow_length = 1.2*self.plot_r_agent
		self.plot_arrow_head_width = self.plot_arrow_width*3
		self.plot_arrow_head_length = self.plot_arrow_head_width


		if self.env_name is 'gridworld':
			print('self.env_ncell: ', self.env_ncell)
			if self.env_ncell > 150:
				exit()