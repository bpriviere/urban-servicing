

import plotter
import datahandler
import glob
import numpy as np 
import matplotlib.pyplot as plt 

from param import Param 

default_param = Param()
desired_controller = 'ctd'

# load sim results 
sim_results = [] # lst of dicts
print('loading sim results...')
for sim_result_dir in glob.glob('../current_results/*'):
	sim_result = datahandler.load_sim_result(sim_result_dir)

	# if desired_controller in sim_result["controller_name"]:
	# 	sim_results.append(sim_result)
	# 	break 
	sim_results.append(sim_result)

print('plotting sim results...')
for sim_result in sim_results:
	controller_name = sim_result["controller_name"]

	full_time = False
	if full_time:
		times = sim_result["times"] 
	else:
		times = [0]

	plotter.sim_plot_over_time(controller_name,sim_result,times)
	
	
plotter.plot_cumulative_reward(sim_results)
plotter.plot_q_error(sim_results)

print('saving and opening results...')
plotter.save_figs(default_param.plot_fn)
plotter.open_figs(default_param.plot_fn)