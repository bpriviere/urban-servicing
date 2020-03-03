
# standard packages
from collections import namedtuple
import numpy as np 

# my packages 
from param import Param
from gridworld import GridWorld
from datahandler import DataHandler
from controller import Controller
import plotter 

class Sim():

	def __init__(self,param,env):
		self.param = param
		self.env = env
		
	def run(self,controller):
			
		results = dict()
		results["times"] = []
		results["rewards"] = []
		for key in self.param.state_keys:
			results[key] = [] 

		env.reset()
		
		for step,time in enumerate(param.sim_times[:-1]):
			print('t = {}/{}'.format(time,param.sim_times[-1]))
			
			observation = env.observe()
			action = controller.policy(observation)
			reward,state = env.step(action)
			
			if param.env_render_on:
				env.render(title='{} at t={}/{}'.format(controller.name,time,param.sim_times[-1]))

			results["times"].append(time)
			results["rewards"].append(reward)
			for key in self.param.state_keys:
				results[key].append(state[key])
				
		return results


if __name__ == '__main__':

	# set random seed
	np.random.seed(0)

	# parameters
	param = Param()

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

	for controller_name, sim_result in sim_results.items():
		plotter.sim_plot_over_time(controller_name,sim_result)

	plotter.plot_sim_rewards(sim_results)

	plotter.save_figs(param.plot_fn)
	plotter.open_figs(param.plot_fn)


