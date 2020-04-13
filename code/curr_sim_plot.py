

import plotter
import datahandler
import glob
import numpy as np 
import matplotlib.pyplot as plt 

from param import Param 

default_param = Param()
full_time = True

# lst of dirs containing folders: sim_result_1
# results_dir = ['../current_results/*']
result_dirs = [
	'../results/2020-04-12_rhc_micro_gridworld/*',
	'../results/2020-04-05_100_agent_gridworld/*',
	]

controller_names = [
	['RHC'],
	['D-TD','H-TD^2','C-TD','Bellman']
	]

# load sim results 
# sim_results = [] # lst of dicts
# print('loading sim results...')
# for result_dir in result_dirs:
# 	for sim_result_dir in glob.glob(result_dir):
# 		sim_result = datahandler.load_sim_result(sim_result_dir)
# 		sim_results.append(sim_result)

sim_results = [] # lst of dicts
print('loading sim results...')
for result_dir, controller_name in zip(result_dirs,controller_names):
	for sim_result_dir in glob.glob(result_dir):
		sim_result = datahandler.load_sim_result(sim_result_dir)
		if sim_result.param["controller_name"] in controller_name:
			sim_results.append(sim_result)


print('plotting sim results...')
for sim_result in sim_results:
	controller_name = sim_result["controller_name"]

	if full_time:
		times = sim_result["times"] 
	else:
		times = [0]

	for timestep,time in enumerate(times):

		count = 0 
		if np.mod(timestep,5) == 0: 
			count += 1 
			print('timestep: ',timestep)
			plotter.render(controller_name,sim_result,timestep)

		if count > 100:
			break 
	
	break

# plotter.macro_plot_number_of_agents(sim_results)
plotter.plot_cumulative_reward(sim_results)
plotter.plot_q_error(sim_results)

print('saving and opening results...')
plotter.save_figs(default_param.plot_fn)
plotter.open_figs(default_param.plot_fn)