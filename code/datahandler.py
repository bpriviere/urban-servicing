
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
			# self.make_gridworld_dataset(env)
			env.make_dataset()
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

	def write_gridworld_dataset(self,env):
		with open("../data/gridworld/customer_requests.npy", "wb") as f:
			np.save(f, env.dataset)
		with open("../data/gridworld/value_fnc_training.npy", "wb") as f:
			np.save(f, env.v0)	
		with open("../data/gridworld/q_values_training.npy", "wb") as f:
			np.save(f, env.q0)

	def load_gridworld_dataset(self,env):
		f = "../data/gridworld/customer_requests.npy"
		env.dataset = np.load(f)
		f = "../data/gridworld/value_fnc_training.npy"
		env.v0 = np.load(f)
		f = "../data/gridworld/q_values_training.npy"
		env.q0 = np.load(f)
	

	def write_sim_results(self, result_dict, filename):
		
		import json
		class NumpyEncoder(json.JSONEncoder):
			def default(self, obj):
				if isinstance(obj, np.ndarray):
					return obj.tolist()
				return json.JSONEncoder.default(self, obj)

		with open('{}.json'.format(filename), 'w') as fp:
			json.dump(result_dict, fp, cls=NumpyEncoder)


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