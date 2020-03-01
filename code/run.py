
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
		
		time_lst,reward_lst,agent_operation_lst,agent_locations_lst,agent_q_values_lst = [],[],[],[],[]
		env.reset()
		for step,time in enumerate(param.sim_times[:-1]):
			print('t = {}/{}'.format(time,param.sim_times[-1]))
	
			if param.env_render_on:
				env.render(title='{} at t={}/{}'.format(controller.name,time,param.sim_times[-1]))

			observation = env.observe()
			action = controller.policy(observation)
			reward,agent_state = env.step(action)
			
			time_lst.append(time)
			reward_lst.append(reward)
			agent_operation_lst.append(agent_state.agent_operation)
			agent_locations_lst.append(agent_state.agent_locations)
			agent_q_values_lst.append(agent_state.agent_q_values)

		if not param.env_render_on:
			env.render()

		times = np.asarray(time_lst)
		rewards = np.asarray(reward_lst)
		agent_operations = np.asarray(agent_operation_lst)
		agent_locations = np.asarray(agent_locations_lst)
		agent_q_values = np.asarray(agent_q_values_lst)

		return times,rewards,agent_operations,agent_locations,agent_q_values


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

	# controller/sim 
	SimResult = namedtuple('SimResult',['name','times','rewards','agent_operation','agent_locations','agent_q_values'])	
	sim_results = []
	for controller_name in param.controllers:

		# policy 
		print('Controller: {}'.format(controller_name))
		controller = Controller(param,env,controller_name)

		# simulation 
		sim = Sim(param,env)
		sim_result = SimResult._make((controller_name,)+sim.run(controller))
		sim_results.append(sim_result)


	for sim_result in sim_results:
		plotter.plot_distribution_over_time(sim_result, env)

	plotter.plot_sim_rewards(sim_results)

	plotter.save_figs(param.plot_fn)
	plotter.open_figs(param.plot_fn)


