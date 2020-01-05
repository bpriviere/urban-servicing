
import numpy as np

# todo install packages 
# import pandas as pd
# import requests

from gridworld import GridWorld
from param import Param 
import utilities


np.random.seed(0)

class DataHandler:
	def __init__(self,param):
		self.param = param 
		

	def make_dataset(self,env):
		if env.name is 'gridworld':
			self.make_gridworld_dataset(env)
			return 
		elif env.name is 'chicago':
			self.make_chicago_dataset(env)
			return 


	def write_dataset(self,env):
		if env.name is 'gridworld':
			return self.write_gridworld_dataset(env)
		elif env.name is 'chicago':
			return self.write_chicago_dataset(env)


	def load_dataset(self,env):
		if self.param.env_name is 'gridworld':
			return self.load_gridworld_dataset(env)
		elif self.param.env_name is 'chicago':
			return self.load_chicago_dataset(env)


	def make_gridworld_dataset(self,env):
		# make dataset that will be used for training and testing
		# training time: [tf_train,0]
		# testing time:  [0,tf_sim]

		tf_train = int(self.param.n_training_data/self.param.n_customers_per_time)
		tf_sim = max(1,int(self.param.sim_times[-1]))

		# 'move' gaussians around for full simulation time 
		env.run_cm_model()

		# training dataset part 
		dataset = []
		customer_time_array_train = np.arange(-tf_train,0,1,dtype=int)
		for time in customer_time_array_train:
			for customer in range(self.param.n_customers_per_time):
				time_of_request = time + np.random.random()
				x_p,y_p = env.sample_cm(0)
				x_d,y_d = utilities.random_position_in_world()
				time_to_complete = env.eta(x_p,y_p,x_d,y_d,time)
				dataset.append(np.array([time_of_request,time_to_complete,x_p,y_p,x_d,y_d]))

		# testing dataset part 
		customer_time_array_sim = np.arange(0,tf_sim,1,dtype=int)
		for time in customer_time_array_sim:
			for customer in range(self.param.n_customers_per_time):
				timestep = int(np.floor(time/env.param.sim_dt))
				time_of_request = time + np.random.random()
				x_p,y_p = env.sample_cm(timestep)
				x_d,y_d = utilities.random_position_in_world()
				time_to_complete = env.eta(x_p,y_p,x_d,y_d,time)
				dataset.append(np.array([time_of_request,time_to_complete,x_p,y_p,x_d,y_d]))

		# solve mdp
		dataset = np.array(dataset)
		train_dataset = dataset[dataset[:,0]<0,:]
		v,q = utilities.solve_MDP(env,train_dataset)
		
		self.dataset = dataset
		self.v = v
		self.q = q 
		

	def write_gridworld_dataset(self,env):
		with open("../data/gridworld/customer_requests.npy", "wb") as f:
			np.save(f, self.dataset)
		with open("../data/gridworld/value_fnc_training.npy", "wb") as f:
			np.save(f, self.v)	
		with open("../data/gridworld/q_values_training.npy", "wb") as f:
			np.save(f, self.q)


	def load_gridworld_dataset(self,env):
		f = "../data/gridworld/customer_requests.npy"
		env.dataset = np.load(f)
		f = "../data/gridworld/value_fnc_training.npy"
		env.v0 = np.load(f)
		f = "../data/gridworld/q_values_training.npy"
		env.q0 = np.load(f)
	

	def write_sim_results(self, results):
		pass 


	def make_chicago_dataset(self, fileSpecifierDict):
		# todo: update function 

		def getDateFilter(self, date):
			#month_str = date['month']
			year_str = date['year']
			#date_filter_str = DATE_EXTRACT_YEAR + year_str + \
			#	'%20AND%20' + DATE_EXTRACT_MONTH + month_str
			date_filter_str = DATE_EXTRACT_YEAR + year_str
			return date_filter_str
			
		def getRequestURL(self, desired_fields, filter_condition):
			desired_fields_str = ','.join(key for (key) in 
				sorted(desired_fields, key=desired_fields.get))
			request_url = DATABASE + desired_fields_str + \
				'%20WHERE%20' + filter_condition
			print(request_url)
			return request_url
			
		def makeDataFile(self, request_url, file_format, filename ):
			resp = requests.get(url=request_url) 
			data = resp.json() 
			df = pd.DataFrame(data)
			df = df[[key for key in sorted(file_format, key=file_format.get)]]
			df = df.dropna()
			filename = 'training_data_raw.csv'
			df.to_csv(filename, header=False, index=False)
			return 0

		DATE_EXTRACT_YEAR = 'date_extract_y(trip_start_timestamp)='
		DATE_EXTRACT_MONTH = 'date_extract_m(trip_start_timestamp)='
		DATABASE = 'https://data.cityofchicago.org/resource/wrvz-psew.json?$query=SELECT%20'			

		date_filter_str = self.getDateFilter( fileSpecifierDict['filter_conditions'] )
		request_url_str = self.getRequestURL( fileSpecifierDict['desired_fields'], date_filter_str )
		self.makeDataFile( request_url_str, fileSpecifierDict['format_desired_fields'], fileSpecifierDict['filename'] )


	def load_chicago_dataset(self):
		pass 