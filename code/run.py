
# standard packages
import numpy as np 
import time as time_pkg
import os 
import glob
import shutil

# my packages 
from param import Param
from gridworld import GridWorld
from citymap import CityMap

from controller import Controller

import datahandler
import plotter 

def run_instance(param):
	# runs sim with given parameters for different controllers and different trials and writes to results directory 

	# init environment 
	if param.env_name in 'gridworld':
		env = GridWorld(param)
	elif param.env_name in 'citymap':
		env = CityMap(param)
	else:
		exit('env_name not recognized: ', param.env_name)


	# run sim 
	for i_trial in range(param.n_trials):

		# init datasets
		if param.make_dataset_on:
			print('   making dataset...')
			train_dataset, test_dataset = datahandler.make_dataset(env)
			datahandler.write_dataset(env, train_dataset, test_dataset)
		print('   loading dataset...')
		datahandler.load_dataset(env)

		# initial condition
		s0 = env.get_s0()

		for (dispatch,task_assignment) in param.controller_names:
			controller = Controller(param,env,dispatch,task_assignment)
		
			# sim 
			sim_result = sim(param,env,controller,s0)
		
			# write results
			case_count = len(glob.glob('../current_results/*')) + 1
			results_dir = '../current_results/sim_result_{}'.format(case_count)
			datahandler.write_sim_result(sim_result, results_dir)
	return 

def sim(param,env,controller,s0):
	# outputs:
	# 	- dictionary with all state variables for all time, plus rewards, times, runtime, controller_name . 

	sim_result = dict()
	sim_result["times"] = []
	sim_result["rewards"] = []
	sim_result["param"] = param.to_dict()
	for key in param.state_keys:
		sim_result[key] = [] 

	sim_result["sim_start_time"] = time_pkg.time()
	sim_result["controller_name"] = controller.name 
	
	env.reset(s0)
	print('   running sim with {}...'.format(controller.name))	
	for step,time in enumerate(param.sim_times[:-1]):
		print('      t = {}/{}'.format(step,len(param.sim_times)))
		
		observation = env.observe()
		action = controller.policy(observation)
		reward,state = env.step(action)

		if param.env_render_on:
			env.render(title='{} at t={}/{}'.format(controller.name,time,param.sim_times[-1]))

		sim_result["times"].append(time)
		sim_result["rewards"].append(reward)
		for key in param.state_keys:
			sim_result[key].append(state[key])

	sim_result["total_reward"] = sum(sim_result["rewards"])
	sim_result["sim_end_time"] = time_pkg.time()
	sim_result["sim_run_time"] = sim_result["sim_end_time"] - sim_result["sim_start_time"]

	return sim_result


if __name__ == '__main__':


	default_param = Param()
	macro_sim_on = False

	# clean results directory
	current_results_dir = '../current_results/*'
	if not os.path.exists('../current_results/*'):
		os.makedirs('../current_results/*',exist_ok=True)
	for old_sim_result_dir in glob.glob(current_results_dir):
		shutil.rmtree(old_sim_result_dir)
		
	# macro sim 
	if macro_sim_on: 
		
		varied_parameter_dict = dict()
		# varied_parameter_dict["env_dx"] = [0.1,0.3,0.5,1.0] 
		varied_parameter_dict["ni"] = [5,10] #10,50,100,150]
		controller_names = default_param.controller_names

		for varied_parameter, varied_parameter_values in varied_parameter_dict.items():
			for varied_parameter_value in varied_parameter_values:
				curr_param = Param()
				setattr(curr_param,varied_parameter,varied_parameter_value)
				curr_param.update()
				run_instance(curr_param)

	# micro sim 
	else:
		if default_param.env_name in 'gridworld':
			default_param.update()
		run_instance(default_param)

	# load sim results 
	sim_results = [] # lst of dicts
	print('loading sim results...')
	for sim_result_dir in glob.glob(current_results_dir):
		sim_results.append(datahandler.load_sim_result(sim_result_dir))

	# plotting 
	print('plotting sim results...')

	if macro_sim_on:
	# 	- plot parameter variation 
		if varied_parameter == "env_dx":
			plotter.plot_runtime_vs_state_space(sim_results)
		elif varied_parameter == "ni":
			plotter.plot_runtime_vs_number_of_agents(sim_results)

	else:
		# 	- plot sim 
		if default_param.plot_sim_over_time:
			for sim_result in sim_results:
				controller_name = sim_result["controller_name"]
				plotter.sim_plot_over_time(controller_name,sim_result)
		else:
			for sim_result in sim_results:
				import matplotlib.pyplot as plt 
				timestep = 0 
				fig = plt.figure()
				if sim_result["param"]["env_name"] in 'citymap':
					city_boundary = plotter.make_city_boundary(sim_result["param"])
					plotter.sim_plot(sim_result["controller_name"],sim_result,timestep,fig=fig,city_boundary=city_boundary)
				else:
					plotter.sim_plot(sim_result["controller_name"],sim_result,timestep,fig=fig)
				break

		# 	- plot reward
		plotter.plot_cumulative_reward(sim_results)

		# 	- plot q value estimation 
		for controller_name in default_param.controller_names:
			if 'bellman' in controller_name:
				plotter.plot_q_error(sim_results)
				break 

		

	print('saving and opening figs...')
	plotter.save_figs(default_param.plot_fn)
	plotter.open_figs(default_param.plot_fn)
