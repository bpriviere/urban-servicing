
# standard
from numpy.random import random
from collections import namedtuple
from mdptoolbox.mdp import PolicyIterationModified
import numpy as np 

# my package 
from agent import Agent, Service
import plotter 
import utilities 

class GridWorld():
	def __init__(self,param):
		self.param = param
		self.name = param.env_name 
		self.timestep = 0
		self.observation = []
		self.init_agents()

		# for plots
		# self.customer_history = [] # list of xy pos of customer locations


	def init_random_variables(self):
		self.eta_w = random(size=(self.param.env_ncell,self.param.env_ncell))*(self.param.wmax-self.param.wmin)+self.param.wmin
		self.eta_phi = random(size=(self.param.env_ncell,self.param.env_ncell))*(self.param.phimax-self.param.phimin)+self.param.phimin
		self.c_w = random(size=(self.param.env_nx,self.param.env_ny))*(self.param.wmax-self.param.wmin)+self.param.wmin
		self.c_phi = random(size=(self.param.env_nx,self.param.env_ny))*(self.param.phimax-self.param.phimin)+self.param.phimin
		return [self.eta_w,self.eta_phi,self.c_w,self.c_phi]


	def init_agents(self):

		# initialize list of agents  
		self.agents = []
		for i in range(self.param.ni):
			x,y = utilities.random_position_in_world()
			self.agents.append(Agent(i,x,y))


	def render(self):
		time=self.param.sim_times[self.timestep]

		fig,ax = plotter.make_fig()
		ax.set_xticks(self.param.env_x)
		ax.set_yticks(self.param.env_y)
		ax.set_xlim(self.param.env_xlim)
		ax.set_ylim(self.param.env_xlim)
		ax.set_aspect('equal')
		ax.set_title('t={}/{}'.format(time,self.param.sim_times[-1]))
		ax.grid(True)

		# state space
		for agent in self.agents:
			color = self.param.plot_agent_mode_color[agent.mode]
			plotter.plot_circle(agent.x,agent.y,self.param.plot_r_agent,fig=fig,ax=ax,color=color)
			plotter.plot_dashed(agent.x,agent.y,self.param.r_comm,fig=fig,ax=ax,color=color)

			if agent.mode == 1:
				plotter.plot_rectangle(agent.service.x_p, agent.service.y_p,\
					self.param.plot_r_customer,fig=fig,ax=ax,color=self.param.plot_customer_color[agent.mode])
			elif agent.mode == 2:
				plotter.plot_rectangle(agent.service.x_d, agent.service.y_d,\
					self.param.plot_r_customer,fig=fig,ax=ax,color=self.param.plot_customer_color[agent.mode])

		for service in self.observation:
			print('service: ', service)
			plotter.plot_rectangle(service.x_p,service.y_p,self.param.plot_r_customer,fig=fig,ax=ax,\
				color=self.param.plot_customer_color[0],angle=45)

		# make customer distribution
		if False:
			im_customer = self.customer_distribution_matrix(time)
			fig,ax = plotter.make_fig()
			ax.set_title('Customer Distribution')
			ax.set_xticks(self.param.env_x)
			ax.set_yticks(self.param.env_y)
			ax.set_xlim(self.param.env_xlim)
			ax.set_ylim(self.param.env_xlim)
			ax.set_aspect('equal')
			ax.grid(True)		
			axim=ax.imshow(im_customer,cmap='gray_r',vmin=0,vmax=1,\
				extent=[self.param.env_xlim[0],self.param.env_xlim[1],self.param.env_ylim[0],self.param.env_ylim[1]])
			fig.colorbar(axim)

			# make agent distribution
			im_agent = np.zeros((self.param.env_nx,self.param.env_ny))
			for agent in self.agents:
				xcell = np.where( self.param.env_x < agent.x)[0][-1]
				ycell = np.where( self.param.env_ylim[1]-self.param.env_y > agent.y)[0][-1]
				im_agent[ycell,xcell] += 1

			im_agent /= len(self.agents)
			fig,ax = plotter.make_fig()
			ax.set_title('Agent Distribution')
			ax.set_xticks(self.param.env_x)
			ax.set_yticks(self.param.env_y)
			ax.set_xlim(self.param.env_xlim)
			ax.set_ylim(self.param.env_xlim)	
			ax.set_aspect('equal')
			ax.grid(True)
			axim=ax.imshow(im_agent,cmap='gray_r',vmin=0,vmax=1,\
				extent=[self.param.env_xlim[0],self.param.env_xlim[1],self.param.env_ylim[0],self.param.env_ylim[1]])
			fig.colorbar(axim)


	def observe(self):
		t0 = self.param.sim_times[self.timestep]
		t1 = self.param.sim_times[self.timestep+1]
		# print('t0: ',t0)
		# print('t1: ',t1)
		# print('self.dataset: ',self.dataset)
		# print('self.dataset[:,0]:',self.dataset[:,0])
		idxs = np.multiply(self.dataset[:,0] >= t0, self.dataset[:,0] < t1, dtype=bool)
		customer_requests = self.dataset[idxs,:]
		for i in range(customer_requests.shape[0]):
			self.observation.append(Service(customer_requests[i,:]))
		return self.observation

	def step(self,actions):

		time = self.param.sim_times[self.timestep]
		wait_time = 0 
		for i_agent, agent in enumerate(self.agents):
			action = actions[i_agent]
			wait_time += agent.step(action,self)
		reward = -wait_time 
		
		AgentState = namedtuple('AgentState',['agent_operation','agent_locations','agent_q_values'])
		agent_operation = np.empty(self.param.ni)
		agent_locations = np.empty((self.param.ni,2))
		agent_q_values = np.empty((self.param.ni,self.param.nq))
		for agent in self.agents:
			agent_operation[agent.i] = agent.mode
			agent_locations[agent.i,:] = [agent.x,agent.y]
			agent_q_values[agent.i,:] = agent.v

		agent_state = AgentState._make((agent_operation,agent_locations,agent_q_values))

		self.timestep += 1
		
		return reward, agent_state

	def eta_cell(self,i,j,t):
		# expected time of arrival between two states (i,j), at given time (t)
		x_i,y_i = utilities.cell_index_to_coordinate(i)
		x_j,y_j = utilities.cell_index_to_coordinate(j)
		return self.eta(x_i,y_i,x_j,y_j,t)

	def eta(self,x_i,y_i,x_j,y_j,t):
		# expected time of arrival between two locations ([x_i,y_i],[x_j,y_j]), at given time (t)
		i = utilities.coordinate_to_cell_index(x_i,y_i)
		j = utilities.coordinate_to_cell_index(x_j,y_j)

		# print('x_i,y_i: {}, {}'.format(x_i,y_i))
		# print('x_j,y_j: {}, {}'.format(x_j,y_j))
		# print('i,j: {}, {}: '.format(i,j))
		
		d_ij = np.linalg.norm([x_i-x_j,y_i-y_j])
		tau = t/self.param.sim_times[-1]
		dist = d_ij + np.sin(self.eta_w[i,j]*tau+self.eta_phi[i,j]) + 1
		return dist/self.param.average_taxi_speed


	def customer_distribution_matrix(self,t):
		tau = t/self.param.sim_times[-1]
		c = np.sin(self.c_w*tau + self.c_phi) + 1
		c = c/np.sum(np.sum(c))
		return c