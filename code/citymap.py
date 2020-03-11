
# standard
from numpy.random import random
from collections import namedtuple
from mdptoolbox.mdp import PolicyIterationModified,ValueIteration
import numpy as np 
import shapefile 
from shapely.geometry import shape 
from shapely.ops import unary_union
from shapely.geometry import Point
import matplotlib.pyplot as plt 

# my package 
from env import Env
from helper_classes import Gaussian, Agent, Service, Dispatch, Empty, CustomerModel
import plotter

class CityMap(Env):
	def __init__(self,param):
		super().__init__(param)
		self.init_map()

		self.dbg_utilities()

	def dbg_utilities(self):

		# todo... 
		# 	xy_cell_index_to_cell_index
		# 	cell_index_to_xy_cell_index
		# 	coordinate_to_xy_cell_index
		# 	random_position_in_cell
		# 	environment_barrier
		# done! 
		# 	random_position_in_world
		# 	coordinate_to_cell_index
		# 	cell_index_to_cell_coordinate


		fig,ax=plt.subplots()
		for i in range(5):

			i_cell = np.random.randint(self.param.env_ncell)
			x_cell,y_cell = self.cell_index_to_xy_cell_index(i_cell)
			print('i_cell',i_cell)
			print('x_cell,y_cell',x_cell,y_cell)
			while not self.valid_cells[x_cell,y_cell]:
				i_cell = np.random.randint(self.param.env_ncell)
				x_cell,y_cell = self.cell_index_to_xy_cell_index(i_cell)

			x,y = self.random_position_in_cell(i_cell)
			print(x,y)

			i_cell = self.coordinate_to_cell_index(x,y)
			print('i_cell',i_cell)

			self.print_map(fig=fig,ax=ax)
			ax.plot([x, x_cell],[y,y_cell],color='green')

		plt.show()

		exit()
		
	def init_map(self):
		print('init map...')

		# read shapefile 
		shp = shapefile.Reader(self.param.shp_path)
		shapes = [shape(polygon) for polygon in shp.shapes()]

		# union shape
		print('   merge polygons in mapfile...')
		union = unary_union(shapes)
		if union.geom_type == 'MultiPolygon':
			self.city_polygon = union[0]			
			for polygon in union:
				if polygon.area > self.city_polygon.area:
					self.city_polygon = polygon
		else:
			self.city_polygon = union

		# make boundary
		x,y = self.city_polygon.exterior.coords.xy
		self.city_boundary = np.asarray([x,y]).T # [npoints x 2]

		# make grid 
		city_bounds = self.city_polygon.bounds # [xmin,ymin,xmax,ymax]
		eps = 0.001
		self.param.env_xlim = [self.city_polygon.bounds[0]-eps,self.city_polygon.bounds[2]+eps]
		self.param.env_ylim = [self.city_polygon.bounds[1]-eps,self.city_polygon.bounds[3]+eps]

		# update parameters with xlim and ylim 
		self.param.update()

		# label valid cells where valid cells are either (inclusive OR)
		# 	- center is in the self.city_polygon  
		# 	- contain an element of the boundary 
		self.valid_cells = np.zeros((len(self.param.env_x[:-1]),len(self.param.env_y[:-1])),dtype=bool)
		
		# x,y are bottom left hand corner of cell 
		for i_x,x in enumerate(self.param.env_x[:-1]):
			for i_y,y in enumerate(self.param.env_y[:-1]):

				cell_center = Point((x + self.param.env_dx/2, y + self.param.env_dy/2))
				if cell_center.within(self.city_polygon): 
					self.valid_cells[i_x,i_y] = True

				else:
					for point in self.city_boundary:
						if self.point_in_cell(point,x,y):
							self.valid_cells[i_x,i_y] = True
							break # out of points-in-city-boundary loop


	def point_in_cell(self,point,x,y):
		# input
		# 	- point is x,y point 
		# 	- x,y is bottom left point of grid cell 
		return (point[0] > x and point[0] < x + self.param.env_dx) and \
			(point[1] > y and point[1] < y + self.param.env_dy)


	def print_map(self,fig=None,ax=None):
		
		ax.plot(self.city_boundary[:,0],self.city_boundary[:,1],marker='.')
		ax.set_xticks(self.param.env_x)
		ax.set_yticks(self.param.env_y)
		ax.set_xlim([self.param.env_xlim[0],self.param.env_xlim[1]])
		ax.set_ylim([self.param.env_ylim[0],self.param.env_ylim[1]])
		ax.grid(True)

		for i_x,x in enumerate(self.param.env_x[:-1]):
			for i_y,y in enumerate(self.param.env_y[:-1]):
				if self.valid_cells[i_x,i_y]:
					ax.plot(x,y,color='red',marker='.')
		
	# ----- plotting -----
	def get_curr_im_value(self):
		# value_fnc_ims = np.zeros((self.param.ni,self.param.env_nx,self.param.env_ny))
		# return value_fnc_ims
		pass
		
	def get_curr_im_gmm(self):
		# im is [nx,ny] where im[0,0] is bottom left
		# return im 
		pass

	def get_curr_im_agents(self):
		# im_agent = np.zeros((self.param.env_nx,self.param.env_ny))
		# return im_agent 
		pass
		
	def get_curr_im_free_agents(self):
		# im_agent = np.zeros((self.param.env_nx,self.param.env_ny))
		# return im_agent
		pass
		
	def get_curr_customer_locations(self):		
		# return customers_location (numpy array)
		pass
		
	def get_agents_int_action(self,actions):
		# pass in list of objects
		# pass out list of integers 
		int_actions_lst = []
		# for i,agent in enumerate(self.agents):
		# 	action = actions[i]
		for agent,action in actions:
			if isinstance(action,Dispatch): 
				s = self.coordinate_to_cell_index(agent.x,agent.y)
				sp = self.coordinate_to_cell_index(action.x,action.y)
				int_a = self.s_sp_to_a(s,sp)
				int_actions_lst.append(int_a)
			elif isinstance(action,Service) or isinstance(action,Empty):
				int_actions_lst.append(-1)
			else:
				exit('get_agents_int_action type error')
		return np.asarray(int_actions_lst)

	def get_agents_vec_action(self,actions):
		
		agents_vec_action = np.zeros((self.param.ni,2))
		for agent,action in actions:
			if isinstance(action,Dispatch):
				# agents_vec_action[i,0] = action.x - agent.x
				# agents_vec_action[i,1] = action.y - agent.y
				sx,sy = self.cell_index_to_cell_coordinate(
					self.coordinate_to_cell_index(agent.x,agent.y))
				agents_vec_action[agent.i,0] = action.x - (sx + self.param.env_dx/2)
				agents_vec_action[agent.i,1] = action.y - (sy + self.param.env_dy/2)

			elif isinstance(action,Service) or isinstance(action,Empty):
				agents_vec_action[agent.i,0] = 0
				agents_vec_action[agent.i,1] = 0
			else:
				print(action)
				exit('get_agents_vec_action type error')
		return agents_vec_action

	def get_curr_ave_vec_action(self,locs,vec_action):
		# locs is (ni,2)
		# vec_actions is (ni,2)
		ni = locs.shape[0]
		im_a = np.zeros((self.param.env_nx,self.param.env_ny,2))
		count = np.zeros((self.param.env_nx,self.param.env_ny,1))
		for i in range(ni):
			idx_x,idx_y = self.coordinate_to_xy_cell_index(locs[i][0],locs[i][1])
			im_a[idx_x,idx_y,:] += vec_action[i][:]
			count[idx_x,idx_y] += 1

		idx = np.nonzero(count)
		# im_a[idx] = (im_a[idx].T/count[idx]).T
		im_a[idx] = im_a[idx]/count[idx]
		return im_a

	# 'cell_index' : element of [0,...,env_ncell]
	# 'cell_coordinate' : (x,y) coordinates of bottom left corner of cell 
	# 'xy_cell_index' : (i_x,i_y) indices corresponding to elements of env_x, env_y
	# 'coordinate' : free (x,y) coordinate, not constrained by being on gridlines

	def cell_index_to_cell_coordinate(self,i):
		x = self.param.env_dx*np.remainder(i,self.param.env_nx) + self.param.env_x[0]
		y = self.param.env_dy*np.floor_divide(i,self.param.env_nx) + self.param.env_y[0]
		return x,y


	def xy_cell_index_to_cell_index(self,i_x,i_y):
		i = i_y*len(self.param.env_x) + i_x
		return int(i) 


	def cell_index_to_xy_cell_index(self,i):
		x,y = self.cell_index_to_cell_coordinate(i)
		i_x,i_y = self.coordinate_to_xy_cell_index(x,y)
		return i_x,i_y


	def coordinate_to_xy_cell_index(self,x,y):
		i = self.coordinate_to_cell_index(x,y)
		x,y = self.cell_index_to_cell_coordinate(i)
		i_x = np.where(self.param.env_x == x)[0][0] 
		i_y = np.where(self.param.env_y == y)[0][0] 
		return int(i_x),int(i_y)


	def coordinate_to_cell_index(self,x,y):
		i_x = np.where(self.param.env_x <= x)[0][-1]
		i_y = np.where(self.param.env_y <= y)[0][-1]
		i = self.xy_cell_index_to_cell_index(i_x,i_y)
		return int(i)


	def random_position_in_cell(self,i):
		x,y = self.cell_index_to_cell_coordinate(i)
		x = self.param.env_dx*random() + x
		y = self.param.env_dy*random() + y
		while not Point((x,y)).within(self.city_polygon):
			x,y = self.cell_index_to_cell_coordinate(i)
			x = self.param.env_dx*random() + x
			y = self.param.env_dy*random() + y
		return x,y


	def random_position_in_world(self):
		x = random()*(self.param.env_xlim[1] - self.param.env_xlim[0]) + self.param.env_xlim[0]
		y = random()*(self.param.env_ylim[1] - self.param.env_ylim[0]) + self.param.env_ylim[0]
		while not Point((x,y)).within(self.city_polygon):
			x = random()*(self.param.env_xlim[1] - self.param.env_xlim[0]) + self.param.env_xlim[0]
			y = random()*(self.param.env_ylim[1] - self.param.env_ylim[0]) + self.param.env_ylim[0]
		return x,y 


	def environment_barrier(self,p):
		eps = 1e-16
		x = np.clip(p[0],self.param.env_xlim[0]+eps,self.param.env_xlim[1]-eps)
		y = np.clip(p[1],self.param.env_ylim[0]+eps,self.param.env_ylim[1]-eps)
		return x,y

	def get_MDP_P(self):
		# P in AxSxS 
		P = np.zeros((self.param.env_naction,self.param.env_ncell,self.param.env_ncell))

		for s in range(self.param.env_ncell):

			x,y = self.cell_index_to_cell_coordinate(s)

			# print('s: ',s)
			# print('x: ',x)
			# print('y: ',y)
			
			# 'empty' action  
			P[0,s,s] = 1.

			# 'right' action
			if not x == self.param.env_x[-1]:
				P[1,s,s+1] = 1.
			else:
				P[1,s,s] = 1.

			# 'top' action
			if not y == self.param.env_y[-1]:
				next_s = self.coordinate_to_cell_index(x,y+self.param.env_dy)
				P[2,s,next_s] = 1.
			else:
				P[2,s,s] = 1.

			# 'left' action
			if not x == self.param.env_x[0]:
				P[3,s,s-1] = 1.
			else:
				P[3,s,s] = 1.			

			# 'down' action
			if not y == self.param.env_y[0]:
				next_s = self.coordinate_to_cell_index(x,y-self.param.env_dy)
				P[4,s,next_s] = 1. 
			else:
				P[4,s,s] = 1.

			# print('P[:,s,:]:', P[:,s,:])
		# exit()

		return P  