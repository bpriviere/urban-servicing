
import numpy as np
import os
import shutil
import json
import glob
import pandas as pd
from sodapy import Socrata
from datetime import datetime,timedelta


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self,obj)

def make_dataset(env):

	if env.name is 'gridworld':
		return env.make_dataset()
	elif env.name is 'citymap':
		return make_citymap_dataset(env)

def write_dataset(env, train_dataset, test_dataset):

	current_data_dir = '../current_data/*'
	if not os.path.exists('../current_data/*'):
		os.makedirs('../current_data/*',exist_ok=True)
	for old_data_dir in glob.glob(current_data_dir):
		shutil.rmtree(old_data_dir)

	datadir = "../current_data/{}".format(env.name)
	if not os.path.exists(datadir):
		os.makedirs(datadir,exist_ok=True)

	with open("{}/train_dataset.npy".format(datadir), "wb") as f:
		np.save(f, train_dataset)
	with open("{}/test_dataset.npy".format(datadir), "wb") as f:
		np.save(f, test_dataset)

def load_dataset(env):

	f_train = "../current_data/{}/train_dataset.npy".format(env.name)
	f_test = "../current_data/{}/test_dataset.npy".format(env.name)

	env.train_dataset = np.load(f_train)
	env.test_dataset = np.load(f_test)
	env.dataset = np.vstack((env.train_dataset,env.test_dataset))

	print('   train_dataset.shape: ', env.train_dataset.shape)
	print('   test_dataset.shape: ', env.test_dataset.shape)

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

	summary_results_fn = '{}/summary_results.json'.format(sim_result_dir)
	summary_results_dict = dict()
	with open(summary_results_fn, 'r') as j:
		summary_results_dict = json.loads(j.read())
	for key,value in summary_results_dict.items():
		sim_result[key] = value

	for file in glob.glob(sim_result_dir + '/*.npy'):
		base = file.split("/")[-1]
		base = base.split(".npy")[0]
		value = np.load(file,allow_pickle=True)
		if base == "controller_name":
			value = str(value)
		sim_result[base] = value

	return sim_result
	

def make_citymap_dataset(env):

	param = env.param
	# time increment 
	delta_minute = 15
	# dilute data by factor
	stepsize = 1

	train_start = datetime(param.train_start_year, 
		param.train_start_month, 
		param.train_start_day, 
		param.train_start_hour, 
		param.train_start_minute, 
		param.train_start_second, 
		param.train_start_microsecond)
	train_end = datetime(param.train_end_year, 
		param.train_end_month, 
		param.train_end_day, 
		param.train_end_hour, 
		param.train_end_minute, 
		param.train_end_second, 
		param.train_end_microsecond)
	test_start = datetime(param.test_start_year, 
		param.test_start_month, 
		param.test_start_day, 
		param.test_start_hour, 
		param.test_start_minute, 
		param.test_start_second, 
		param.test_start_microsecond)
	test_end = datetime(param.test_end_year, 
		param.test_end_month, 
		param.test_end_day, 
		param.test_end_hour, 
		param.test_end_minute, 
		param.test_end_second, 
		param.test_end_microsecond) 

	train_dataset = make_citymap_dataset_instance(train_start,train_end,delta_minute,10)
	test_dataset = make_citymap_dataset_instance(test_start,test_end,delta_minute,stepsize)

	return train_dataset,test_dataset 

def make_citymap_dataset_instance(datetime_start,datetime_end,delta_minute,stepsize):

	input_timestamp_key = "'%Y-%m-%dT%H:%M:%S'"
	output_timestamp_key = "%Y-%m-%dT%H:%M:%S"
	
	datetime_curr = datetime_start
	datetime_next = datetime_start
	while (datetime_end-datetime_curr).total_seconds() > 0:

		datetime_curr = datetime_next
		datetime_next = datetime_curr + timedelta(minutes=delta_minute)

		timestamp_curr = datetime_curr.strftime(input_timestamp_key) 
		timestamp_next = datetime_next.strftime(input_timestamp_key) 

		filter_condition = "trip_start_timestamp between "+timestamp_curr+" and "+timestamp_next

		print(filter_condition)

		print('   contacting client...')
		client = Socrata("data.cityofchicago.org", None)
		print('   getting data...')
		results = client.get("wrvz-psew", where = filter_condition, limit=5000)
		print('   reading into pandas...')
		results_df = pd.DataFrame.from_records(results)

		# print(results_df)
		nrows = results_df.shape[0]
		if nrows > 0:


			time_of_requests = []
			for trip_start_timestamp in results_df["trip_start_timestamp"]:
				starttime = datetime.strptime(trip_start_timestamp[0:-4], output_timestamp_key).timestamp()
				starttime += np.random.uniform()*60*15
				time_of_requests.append(starttime)

			idx = np.arange(0,nrows,stepsize,dtype=int)
			dataset_i = np.empty((len(idx),6))

			# time of request [s]
			dataset_i[:,0] = np.asarray(time_of_requests)[idx]
			# time to complete [s]
			dataset_i[:,1] = results_df["trip_seconds"][idx]
			# x_p (longitude) [degrees]
			dataset_i[:,2] = results_df["pickup_centroid_longitude"][idx]
			# y_p (latitude) [degrees]
			dataset_i[:,3] = results_df["pickup_centroid_latitude"][idx]
			# x_p (longitude) [degrees]
			dataset_i[:,4] = results_df["dropoff_centroid_longitude"][idx]
			# y_p (latitude) [degrees]
			dataset_i[:,5] = results_df["dropoff_centroid_latitude"][idx]
		
			if 'dataset' in locals():
				dataset = np.vstack((dataset,dataset_i))
			else:
				dataset = dataset_i 

		# print(dataset.shape)

	# dirty data
	print('dirty data shape: ', dataset.shape)

	# clean data
	dataset = dataset[~np.isnan(dataset).any(axis=1)]

	print('final dataset size: ', dataset.shape)

	return dataset
	

