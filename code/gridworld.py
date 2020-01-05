
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
		self.init_cm()

		# x = np.array([-1,-2,-4])
		# softmax_x = utilities.softmax(x)
		# print(x)
		# print(softmax_x)
		# exit()

	def init_agents(self):
		# initialize list of agents  
		self.agents = []
		for i in range(self.param.ni):
			x,y = utilities.random_position_in_world()
			# x,y = [1.5,.75]
			self.agents.append(Agent(i,x,y,self.v0,self.q0))

			print('agent {} initialized at (x,y) = ({},{})'.format(i,x,y))


	def render(self,title=None):
		time=self.param.sim_times[self.timestep]

		fig,ax = plotter.make_fig()
		ax.set_xticks(self.param.env_x)
		ax.set_yticks(self.param.env_y)
		ax.set_xlim(self.param.env_xlim)
		ax.set_ylim(self.param.env_ylim)
		ax.set_aspect('equal')
		if title is None:
			ax.set_title('t={}/{}'.format(time,self.param.sim_times[-1]))
		else:
			ax.set_title(title)
		ax.grid(True)

		# state space
		for agent in self.agents:
			
			color = self.param.plot_agent_mode_color[agent.mode]
			plotter.plot_circle(agent.x,agent.y,self.param.plot_r_agent,fig=fig,ax=ax,color=color)
			
			if agent.i == 0:
				plotter.plot_dashed(agent.x,agent.y,self.param.r_comm,fig=fig,ax=ax,color=color)
		
			# pickup 
			if agent.mode == 1:
				plotter.plot_rectangle(agent.service.x_p, agent.service.y_p,\
					self.param.plot_r_customer,fig=fig,ax=ax,color=self.param.plot_customer_color[agent.mode])
				plotter.plot_line(agent.x,agent.y,agent.service.x_p,agent.service.y_p,fig=fig,ax=ax,color=self.param.plot_customer_color[agent.mode])

			# dropoff 
			elif agent.mode == 2:
				plotter.plot_rectangle(agent.service.x_d, agent.service.y_d,\
					self.param.plot_r_customer,fig=fig,ax=ax,color=self.param.plot_customer_color[agent.mode])
				plotter.plot_line(agent.x,agent.y,agent.service.x_d,agent.service.y_d,fig=fig,ax=ax,color=self.param.plot_customer_color[agent.mode])

			# on dispatch 
			elif agent.mode == 3 and self.param.plot_arrows_on:	
				dx = agent.dispatch.x - agent.x
				dy = agent.dispatch.y - agent.y
				plotter.plot_arrow(agent.x,agent.y,dx,dy,fig=fig,ax=ax,color=self.param.plot_customer_color[agent.mode])

		for service in self.observation:
			print('service: ', service)
			plotter.plot_rectangle(service.x_p,service.y_p,self.param.plot_r_customer,fig=fig,ax=ax,\
				color=self.param.plot_customer_color[0],angle=45)

		if self.param.plot_distribution_error_on:
			
			# customer distribution
			im_customer = self.eval_cm(self.timestep)
			
			# agent distribution 
			im_agent = np.zeros((self.param.env_nx,self.param.env_ny))
			# for agent in self.agents:
			# 	xcell,ycell = utilities.coordinate_to_xy_cell_index(agent.x,agent.y)
			# 	im_agent[xcell,ycell] += 1
			# im_agent /= len(self.agents)

			# error
			im_err = np.abs(im_agent-im_customer)

			# align axis with image coordinate system 
			im_err = im_err.T
			im_err = np.flipud(im_err)

			# plot
			fig,ax = plotter.make_fig()
			ax.set_title('Distribution Error')
			ax.set_xticks(self.param.env_x)
			ax.set_yticks(self.param.env_y)
			ax.set_xlim(self.param.env_xlim)
			ax.set_ylim(self.param.env_ylim)
			ax.set_aspect('equal')
			ax.grid(True)
			axim=ax.imshow(im_err,cmap='gray_r',vmin=0,vmax=1, 
				extent=[self.param.env_xlim[0],self.param.env_xlim[1],self.param.env_ylim[0],self.param.env_ylim[1]])
			fig.colorbar(axim)

		if self.param.plot_value_fnc_on:

			# v = self.agents[0].v
			v = utilities.q_value_to_value_fnc(self.agents[0].q)

			im_v = np.zeros((self.param.env_nx,self.param.env_ny))
			for i in range(self.param.env_ncell):
				i_x,i_y = utilities.cell_index_to_xy_cell_index(i)
				im_v[i_x,i_y] = v[i]

			# print('v: ', v)

			im_v = im_v.T
			im_v = np.flipud(im_v)

			# print('im_v: ', im_v)

			fig,ax = plotter.make_fig()
			ax.set_title('Value Function')
			ax.set_xticks(self.param.env_x)
			ax.set_yticks(self.param.env_y)
			ax.set_xlim(self.param.env_xlim)
			ax.set_ylim(self.param.env_ylim)
			ax.set_aspect('equal')
			ax.grid(True)
			# axim=ax.imshow(im_v,cmap='gray_r',vmin=-1,vmax=0, 
			# 	extent=[self.param.env_xlim[0],self.param.env_xlim[1],self.param.env_ylim[0],self.param.env_ylim[1]])
			axim=ax.imshow(im_v,
				extent=[self.param.env_xlim[0],self.param.env_xlim[1],self.param.env_ylim[0],self.param.env_ylim[1]])
			fig.colorbar(axim)

			# plotter.show_figs()
			# exit()

	def reset(self):
		self.timestep = 0
		self.observation = []
		self.init_agents()


	def observe(self):
		t0 = self.param.sim_times[self.timestep]
		t1 = self.param.sim_times[self.timestep+1]
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
			agent_q_values[agent.i,:] = agent.q

		agent_state = AgentState._make((agent_operation,agent_locations,agent_q_values))

		self.timestep += 1

		return reward, agent_state

	def eta_cell(self,i,j,t):
		# expected time of arrival between two states (i,j), at given time (t)
		x_i,y_i = utilities.cell_index_to_cell_coordinate(i)
		x_j,y_j = utilities.cell_index_to_cell_coordinate(j)
		x_i += self.param.env_dx/2
		x_j += self.param.env_dx/2
		y_i += self.param.env_dy/2
		y_j += self.param.env_dy/2
		return self.eta(x_i,y_i,x_j,y_j,t)

	def eta(self,x_i,y_i,x_j,y_j,t):
		dist = np.linalg.norm([x_i-x_j,y_i-y_j])
		return dist/self.param.taxi_speed


	def init_cm(self):

		class Gaussian:
			def __init__(self,i,x,y,s,v):
				self.i = i
				self.s = s
				self.v = v 
				self.x = np.empty(nt)
				self.y = np.empty(nt)
				self.x[0] = x
				self.y[0] = y
				# print('self.x: ', self.x)
				# print('self.y: ', self.y)

			def move(self,p,timestep):
				self.x[timestep+1] = self.x[timestep] + p[0]
				self.y[timestep+1] = self.y[timestep] + p[1]

			def sample(self,timestep):
				x,y = np.random.normal([self.x[timestep],self.y[timestep]],self.s)
				return x,y

		# make customer model list
		nt = len(self.param.sim_times) 
		cgm_lst = []
		for i in range(self.param.cm_ng):
			# x0,y0 = utilities.random_position_in_world()
			x0,y0 = [self.param.env_x[1]/2, self.param.env_y[-1]/2]
			# print('cgm (x0,y0) = ({},{})'.format(x0,y0))
			cgm_lst.append(
				Gaussian(i,x0,y0,self.param.cm_sigma,self.param.cm_speed))

			print('cgm {} initialized at (x,y) = ({},{})'.format(i,x0,y0))

		self.cgm_lst = cgm_lst


	def sample_cm(self,timestep):
		# sample multimodal gaussian model

		# weight vector 
		w = np.ones((self.param.cm_ng))/self.param.cm_ng
		# sample w 
		i = np.random.choice(self.param.cm_ng,p=w)
		# sample ith gaussian model of cgm_lst
		x,y = self.cgm_lst[i].sample(timestep)
		return x,y


	def move_cm(self,timestep):
		# move gaussians

		dt = self.param.sim_dt 
		for cgm in self.cgm_lst:
			# th = np.random.random()*2*np.pi
			th = 0 
			unit_vec = np.array([np.cos(th),np.sin(th)])
			move = cgm.v*dt*unit_vec
			p = [cgm.x[timestep] + move[0],cgm.y[timestep] + move[1]]
			p = utilities.environment_barrier(p)
			safe_move = [p[0] - cgm.x[timestep], p[1] - cgm.y[timestep]]
			cgm.move(safe_move,timestep)

	def run_cm_model(self):
		for step,t in enumerate(self.param.sim_times[:-1]):
			self.move_cm(step)

	def eval_cm(self,timestep):
		# input: 
		# 	- self : env
		# 	- t : time
		# output: 
		# 	- cm : customer model probability matrix with shape: (env_nx,env_ny), where sum(sum(cm)) = 1 

		# for cgm in self.cgm_lst:
		# 	print('(cgm.x,cgm.y) = ({},{})'.format(cgm.x,cgm.y))

		cm = np.zeros((self.param.env_nx,self.param.env_ny))
		for i in range(self.param.cm_nsample_cm):
			x,y = self.sample_cm(timestep)
			x,y = utilities.environment_barrier([x,y])
			i_x,i_y = utilities.coordinate_to_xy_cell_index(x,y)
			cm[i_x,i_y] += 1

		# normalize
		cm = cm/sum(sum(cm))
		return cm 