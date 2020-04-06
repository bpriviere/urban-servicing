

import plotter
import datahandler
import glob
import numpy as np 
import matplotlib.pyplot as plt 

from param import Param 

default_param = Param()
desired_controller = 'ctd'
desired_control_on = False
full_time = True

# load sim results 
sim_results = [] # lst of dicts
print('loading sim results...')
for sim_result_dir in glob.glob('../current_results/*'):
	sim_result = datahandler.load_sim_result(sim_result_dir)

	if desired_control_on:
		if desired_controller in sim_result["controller_name"]:
			sim_results.append(sim_result)
			break 
	else:
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

# plotter.plot_runtime_vs_number_of_agents(sim_results)
# plotter.plot_totalreward_vs_number_of_agents(sim_results)
plotter.plot_cumulative_reward(sim_results)
plotter.plot_q_error(sim_results)

print('saving and opening results...')
plotter.save_figs(default_param.plot_fn)
plotter.open_figs(default_param.plot_fn)