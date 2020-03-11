

import plotter
import datahandler
import glob

from param import Param 

default_param = Param()

# load sim results 
sim_results = [] # lst of dicts
print('loading sim results...')
for sim_result_dir in glob.glob(default_param.results_dir + '/*'):
	sim_result = datahandler.load_sim_result(sim_result_dir)
	sim_results.append(sim_result)

# plotting 
print('plotting sim results...')
for sim_result in sim_results:
	controller_name = sim_result["controller_name"]
	plotter.sim_plot_over_time(controller_name,sim_result)
	
plotter.plot_cumulative_reward(sim_results)

plotter.save_figs(default_param.plot_fn)
plotter.open_figs(default_param.plot_fn)