
# standard packages
import numpy as np 
import time as time_pkg
import os 
import glob

# my packages 
from param import Param
from gridworld import GridWorld
from datahandler import DataHandler
from controller import Controller
# from utilities import Utility
from plotter import Plotter
# import plotter 


def run_instance(param):
	# runs sim with given parameters for different controllers and different trials and writes to results directory 
	# output:
	# 	- dicts of lsts of dicts

	env = GridWorld(param)
	datahandler = DataHandler(param)
	controller_names = param.controller_names
	plotter = Plotter(param)

	# can we repack this stuff??
	if param.make_dataset_on:
		print('making dataset...')
		datahandler.make_dataset(env)
		datahandler.write_dataset(env)
	print('loading dataset...')
	datahandler.load_dataset(env)

	sim_results_by_controller = dict()
	for controller_name in controller_names:
		sim_results_by_controller[controller_name] = []
		controller = Controller(param,env,controller_name)
		for i_trial in range(param.n_trials):
			sim_result = sim(param,env,controller)
			sim_results_by_controller[controller_name].append(sim_result)
			case_count = len(glob.glob(default_param.results_dir + "/*.json"))
			filename = param.results_dir + '/case{}'.format(case_count)
			datahandler.write_sim_result(sim_result, filename)

			if param.plot_sim_over_time:
				plotter.sim_plot_over_time(controller_name,sim_result)

	return sim_results_by_controller

def sim(param,env,controller):
	# outputs:
	# 	- dictionary with all state variables for all time, plus rewards, times, runtime. 

	sim_result = dict()
	sim_result["times"] = []
	sim_result["rewards"] = []
	sim_result["param"] = param.to_dict()
	for key in param.state_keys:
		sim_result[key] = [] 

	sim_result["sim_start_time"] = time_pkg.time()
	sim_result["controller_name"] = controller.name 
	
	env.reset()
	print('running sim...')	
	for step,time in enumerate(param.sim_times[:-1]):
		print('   t = {}/{}'.format(time,param.sim_times[-1]))
		
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

	varied_parameter_dict = dict()
	varied_parameter_dict["ni"] = [default_param.ni]
	controller_names = default_param.controller_names

	# clean results directory
	if True: 
		for file in glob.glob(default_param.results_dir + "/*.json"):
			os.remove(file)
	
	for varied_parameter, varied_parameter_values in varied_parameter_dict.items():
		for varied_parameter_value in varied_parameter_values:
			curr_param = Param()
			setattr(curr_param,varied_parameter,varied_parameter_value)
			sim_results_by_controller = run_instance(curr_param)

	# plotting 
	plotter = Plotter(default_param)
	fig,ax = plotter.make_fig()
	for controller_name, sim_results in sim_results_by_controller.items():
		plotter.plot_cumulative_reward_w_trials(sim_results,ax=ax,label=controller_name)
	
	ax.legend()
	plotter.save_figs(default_param.plot_fn)
	plotter.open_figs(default_param.plot_fn)