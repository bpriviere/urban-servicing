
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
		Result = namedtuple('Result',['times','rewards','agent_operation','agent_locations','agent_q_values'])
		
		time_lst,reward_lst,agent_operation_lst,agent_locations_lst,agent_q_values_lst = [],[],[],[],[]
		for step,time in enumerate(param.sim_times[:-1]):

			print('t = {}/{}'.format(time,param.sim_times[-1]))
	
			if param.env_render_on:
				env.render()				

			observation = env.observe()
			# print('observation: ', observation)
			action = controller.policy(observation)
			reward,agent_state = env.step(action)
			
			time_lst.append(time)
			reward_lst.append(reward)
			agent_operation_lst.append(agent_state.agent_operation)
			agent_locations_lst.append(agent_state.agent_locations)
			agent_q_values_lst.append(agent_state.agent_q_values)

		self.result = Result._make((time_lst,reward_lst,agent_operation_lst,agent_locations_lst,agent_q_values_lst))



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
		datahandler.make_dataset()
		datahandler.write_dataset()
	
	datahandler.load_dataset(env)

	# controller/sim 	
	for controller_name in param.controllers:

		# policy 
		print('Controller: {}'.format(controller_name))
		controller = Controller(param,env,controller_name)
		
		# simulation 
		sim = Sim(param,env)
		sim.run(controller)	
	
	plotter.save_figs(param.plot_fn)
	plotter.open_figs(param.plot_fn)
