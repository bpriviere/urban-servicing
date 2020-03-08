
import numpy as np
import os
import json
import glob

# todo install packages 
# import pandas as pd
# import requests

from gridworld import GridWorld
from param import Param 
import utilities

class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self,obj)

def make_dataset(env):
	if env.name is 'gridworld':
		env.make_dataset()
		return 
	elif env.name is 'chicago':
		make_chicago_dataset(env)
		return 

def write_dataset(env):
	if env.name is 'gridworld':
		return write_gridworld_dataset(env)
	elif env.name is 'chicago':
		return write_chicago_dataset(env)

def load_dataset(env):
	if env.name is 'gridworld':
		return load_gridworld_dataset(env)
	elif env.env_name is 'chicago':
		return load_chicago_dataset(env)

def write_gridworld_dataset(env):

	datadir = "../data/gridworld"
	
	if not os.path.exists(datadir):
		os.makedirs(datadir,exist_ok=True)

	with open("{}/customer_requests.npy".format(datadir), "wb") as f:
		np.save(f, env.dataset)
	with open("{}/value_fnc_training.npy".format(datadir), "wb") as f:
		np.save(f, env.v0)	
	with open("{}/q_values_training.npy".format(datadir), "wb") as f:
		np.save(f, env.q0)

def load_gridworld_dataset(env):
	f = "../data/gridworld/customer_requests.npy"
	env.dataset = np.load(f)
	f = "../data/gridworld/value_fnc_training.npy"
	env.v0 = np.load(f)
	f = "../data/gridworld/q_values_training.npy"
	env.q0 = np.load(f)

def write_sim_result( sim_result, sim_result_dir):

	if not os.path.exists(sim_result_dir):
		os.makedirs(sim_result_dir)

	# parameters of simulation into a dict 
	param_fn = '{}/param.json'.format(sim_result_dir)
	with open(param_fn, 'w') as fp:
		json.dump(sim_result["param"], fp, cls=NumpyEncoder, indent=2)

	# summary scalar results into a dict 
	summary_results_keys = ['controller_name','total_reward','sim_run_time','sim_start_time','sim_end_time']
	summary_results_keys_dict = dict()
	for state_key in summary_results_keys:
		summary_results_keys_dict[state_key] = sim_result[state_key]
	summary_results_fn = '{}/summary_results.json'.format(sim_result_dir)
	with open(summary_results_fn, 'w') as fp:
		json.dump(summary_results_keys_dict, fp, cls=NumpyEncoder, indent=2)
		
	# long results in npy files 
	other_keys = ['times','rewards']
	for other_key in other_keys:
		with open("{}/{}.npy".format(sim_result_dir,other_key), "wb") as f:
			np.save(f,sim_result[other_key])

	# state results in npy files
	for state_key in sim_result["param"]["state_keys"]:
		with open("{}/{}.npy".format(sim_result_dir,state_key), "wb") as f:
			np.save(f,sim_result[state_key])	


def load_sim_result(sim_result_dir):
	# input directory, output sim_result dict
	sim_result = dict()

	param_fn = '{}/param.json'.format(sim_result_dir)
	with open(param_fn, 'r') as j:
		sim_result["param"] = json.loads(j.read())

	for file in glob.glob(sim_result_dir + '/*.npy'):
		base = file.split("/")[-1]
		base = base.split(".npy")[0]
		value = np.load(file,allow_pickle=True)
		if base == "controller_name":
			value = str(value)
		sim_result[base] = value

	return sim_result
	


# def make_chicago_dataset(self, fileSpecifierDict):
# 	# todo: update function 

# 	def getDateFilter(self, date):
# 		#month_str = date['month']
# 		year_str = date['year']
# 		#date_filter_str = DATE_EXTRACT_YEAR + year_str + \
# 		#	'%20AND%20' + DATE_EXTRACT_MONTH + month_str
# 		date_filter_str = DATE_EXTRACT_YEAR + year_str
# 		return date_filter_str
		
# 	def getRequestURL(self, desired_fields, filter_condition):
# 		desired_fields_str = ','.join(key for (key) in 
# 			sorted(desired_fields, key=desired_fields.get))
# 		request_url = DATABASE + desired_fields_str + \
# 			'%20WHERE%20' + filter_condition
# 		print(request_url)
# 		return request_url
		
# 	def makeDataFile(self, request_url, file_format, filename ):
# 		resp = requests.get(url=request_url) 
# 		data = resp.json() 
# 		df = pd.DataFrame(data)
# 		df = df[[key for key in sorted(file_format, key=file_format.get)]]
# 		df = df.dropna()
# 		filename = 'training_data_raw.csv'
# 		df.to_csv(filename, header=False, index=False)
# 		return 0

# 	DATE_EXTRACT_YEAR = 'date_extract_y(trip_start_timestamp)='
# 	DATE_EXTRACT_MONTH = 'date_extract_m(trip_start_timestamp)='
# 	DATABASE = 'https://data.cityofchicago.org/resource/wrvz-psew.json?$query=SELECT%20'			

# 	date_filter_str = self.getDateFilter( fileSpecifierDict['filter_conditions'] )
# 	request_url_str = self.getRequestURL( fileSpecifierDict['desired_fields'], date_filter_str )
# 	self.makeDataFile( request_url_str, fileSpecifierDict['format_desired_fields'], fileSpecifierDict['filename'] )


# def load_chicago_dataset(self):
# 	pass 