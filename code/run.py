
# standard packages
import numpy as np 
import time as time_pkg

# my packages 
from param import Param
from gridworld import GridWorld
from datahandler import DataHandler
from controller import Controller
# from utilities import Utility
from plotter import Plotter
# import plotter 

class Sim():

	def __init__(self,param,env):
		self.param = param
		self.env = env
		
	def run(self,controller):

		# init results dict
		results = dict()
		results["times"] = []
		results["rewards"] = []
		for key in self.param.state_keys:
			results[key] = [] 

		results["sim_start_time"] = time_pkg.time()

		# sim 
		self.env.reset()
		print('running sim...')	
		for step,time in enumerate(param.sim_times[:-1]):
			print('   t = {}/{}'.format(time,param.sim_times[-1]))
			
			observation = self.env.observe()
			action = controller.policy(observation)
			reward,state = self.env.step(action)
			
			if param.env_render_on:
				env.render(title='{} at t={}/{}'.format(controller.name,time,param.sim_times[-1]))

			results["times"].append(time)
			results["rewards"].append(reward)
			for key in self.param.state_keys:
				results[key].append(state[key])

		results["sim_end_time"] = time_pkg.time()
		results["sim_run_time"] = results["sim_end_time"] - results["sim_start_time"]

		return results


def run_instance(param):

	print('Parameters: {}'.format(param))

	# environment 
	print('Env: {}'.format(param.env_name))
	if param.env_name is 'gridworld':
		env = GridWorld(param)
	else:
		exit('fatal error: param.env_name not recognized')

	# data 
	datahandler = DataHandler(param)
	if param.make_dataset_on:
		print('making dataset...')
		datahandler.make_dataset(env)
		datahandler.write_dataset(env)
	
	print('loading dataset...')
	datahandler.load_dataset(env)

	# sim each controller
	sim_results = dict()
	for controller_name in param.controllers:

		# policy 
		print('Controller: {}'.format(controller_name))
		controller = Controller(param,env,controller_name)

		# simulation 
		sim = Sim(param,env)
		sim_results[controller_name] = sim.run(controller)

	print('writing results...')
	datahandler.write_sim_results(sim_results, param.results_filename)

	return sim_results


if __name__ == '__main__':

	# set random seed
	np.random.seed(0)

	ni_lst = [5,10,15,20,50]
	macro_sim_results = []
	for ni in ni_lst:
		# parameters
		param = Param()
		param.ni = ni
		param.results_filename = param.results_filename + '_{}ni'.format(ni)

		# run 
		sim_results = run_instance(param)
		
		# save param 
		sim_results["param"] = param.to_dict()

		# add to lst 
		macro_sim_results.append(sim_results)

	plotter = Plotter(param)
	plotter.macro_sim_plot(macro_sim_results, ni_lst)
	plotter.save_figs(param.plot_fn)
	plotter.open_figs(param.plot_fn)

	# if param.plot_sim_over_time:
	# 	for controller_name, sim_result in sim_results.items():
	# 		plotter.sim_plot_over_time(controller_name,sim_result)
	# plotter.plot_sim_rewards(sim_results)




