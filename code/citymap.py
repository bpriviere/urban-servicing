
# standard
from numpy.random import random
from collections import namedtuple
from mdptoolbox.mdp import PolicyIterationModified,ValueIteration
import numpy as np 
import shapefile 
from shapely.geometry import shape 
from shapely.ops import unary_union, nearest_points, split
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt 

# my package 
from env import Env
from helper_classes import Gaussian, Agent, Service, Dispatch, Empty, CustomerModel
import plotter

class CityMap(Env):
	def __init__(self,param):
		super().__init__(param)
		self.init_map()

		
	def init_map(self):
		# this fnc 
		# 	- creates city polygon object from shapefile
		# 	- creates city boundary array from shapefile
		# 	- creates cell <-> grid index maps 

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
		# self.param.env_xlim = [self.city_polygon.bounds[0]+eps,self.city_polygon.bounds[2]-eps]
		# self.param.env_ylim = [self.city_polygon.bounds[1]+eps,self.city_polygon.bounds[3]-eps]
		self.param.env_xlim = [self.city_polygon.bounds[0]-eps,self.city_polygon.bounds[2]+eps]
		self.param.env_ylim = [self.city_polygon.bounds[1]-eps,self.city_polygon.bounds[3]+eps]

		# update parameters with xlim and ylim 
		self.param.update()

		# label valid cells where valid cells are either (inclusive OR)
		# 	- center is in the self.city_polygon  
		# 	- contain an element of the boundary 
		self.valid_cells_mask = np.zeros((len(self.param.env_x[:-1]),len(self.param.env_y[:-1])),dtype=bool)
		
		# x,y are bottom left hand corner of cell 
		for i_x,x in enumerate(self.param.env_x[:-1]):
			for i_y,y in enumerate(self.param.env_y[:-1]):

				cell_center = Point((x + self.param.env_dx/2, y + self.param.env_dy/2))
				if cell_center.within(self.city_polygon): 
					self.valid_cells_mask[i_x,i_y] = True

				else:
					for point in self.city_boundary:
						if self.point_in_cell(point,x,y):
							self.valid_cells_mask[i_x,i_y] = True
							break # out of points-in-city-boundary loop

		# kind of awkward 
		self.param.env_ncell = int(sum(sum(self.valid_cells_mask)))
		self.param.nq = self.param.env_ncell*self.param.env_naction

		# make utility maps 
		self.grid_index_to_cell_index_map = np.zeros((self.valid_cells_mask.shape),dtype=int)
		self.cell_index_to_grid_index_map = np.zeros((self.param.env_ncell,2),dtype=int)
		count = 0
		for i_x,x in enumerate(self.param.env_x[:-1]):
			for i_y,y in enumerate(self.param.env_y[:-1]):
				if self.valid_cells_mask[i_x,i_y]:
					self.grid_index_to_cell_index_map[i_x,i_y] = count
					self.cell_index_to_grid_index_map[count,:] = [i_x,i_y]
					count += 1

	def print_map(self,fig=None,ax=None):
		ax.plot(self.city_boundary[:,0],self.city_boundary[:,1],linewidth=1)
		ax.set_xticks(self.param.env_x)
		ax.set_yticks(self.param.env_y)
		ax.set_xlim([self.param.env_xlim[0],self.param.env_xlim[1]])
		ax.set_ylim([self.param.env_ylim[0],self.param.env_ylim[1]])
		ax.grid(True)
		
	# ----- plotting -----	
	def get_curr_im_gmm(self):
		# im is [nx,ny] where im[0,0] is bottom left
		# return im 
		pass
		
	def get_curr_im_free_agents(self):
		# im_agent = np.zeros((self.param.env_nx,self.param.env_ny))
		# return im_agent
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

	# Utility Stuff
		# 'cell_index' : element of [0,...,env_ncell]
		# 'cell_coordinate' : (x,y) coordinates of bottom left corner of cell 
		# 'xy_cell_index' : (i_x,i_y) indices corresponding to elements of env_x, env_y
		# 'coordinate' : free (x,y) coordinate, not constrained by being on gridlines

	def cell_index_to_cell_coordinate(self,i):
		# takes in valid cell index and returns bottom left corner coordinate of cell
		# dim(self.cell_index_to_cell_coordinate_map) = [nvalidcells, 2]
		i_x,i_y = self.cell_index_to_grid_index_map[i,:]
		x,y = self.grid_index_to_coordinate(i_x,i_y)
		return x,y

	def coordinate_to_grid_index(self,x,y):
		# takes in coordinate and returns which i_x,i_y cell it is in
		i_x = np.where(self.param.env_x <= x)[0][-1] # last index where input-x is larger than grid 
		i_y = np.where(self.param.env_y <= y)[0][-1]
		return i_x,i_y

	def coordinate_to_cell_index(self,x,y):
		i_x,i_y = self.coordinate_to_grid_index(x,y)
		i = self.grid_index_to_cell_index_map[i_x,i_y] # i should always be a valid index 
		return i

	def grid_index_to_coordinate(self,i_x,i_y):
		x = self.param.env_x[i_x]
		y = self.param.env_y[i_y]
		return x,y

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

	def point_in_cell(self,point,x,y):
		# input
		# 	- point is x,y point 
		# 	- x,y is bottom left point of grid cell 
		# output
		# 	- bool: true if point in cell 
		return (point[0] > x and point[0] < x + self.param.env_dx) and \
			(point[1] > y and point[1] < y + self.param.env_dy)

	def environment_barrier(self,p):

		point = Point(p)

		if point.within(self.city_polygon):
			x = p[0]
			y = p[1]
		else: 
			points = nearest_points(self.city_polygon, point)
			x = points[0].x
			y = points[0].y
		return x,y

	def eta(self,x_i,y_i,x_j,y_j):
		dist = np.linalg.norm([x_i-x_j,y_i-y_j])
		return dist/self.param.taxi_speed		

	def dbg_utilities(self):

		# todo... 
			# get_MDP_P
		# done...
			# environment_barrier
			# random_position_in_cell
			# point_in_cell
			# cell_index_to_cell_coordinate
			# random_position_in_world
			# coordinate_to_grid_index
			# grid_index_to_coordinate
			# coordinate_to_cell_index

		fig,ax=plt.subplots()
		self.print_map(fig=fig,ax=ax)
		plt.show()
		exit()		

		# for i in range(5):
			# test 1:
				# - pick a random coordinate 
				# - find its grid index 
				# - plot its grid coordinate
			# x,y = self.random_position_in_world()
			# i_x,i_y = self.coordinate_to_grid_index(x,y)
			# x_cell,y_cell = self.grid_index_to_coordinate(i_x,i_y)
			# ax.plot([x, x_cell],[y,y_cell],color='green')

			# test 2:
				# - pick a random coordinate 
				# - find its cell index 
				# - plot its cell coordinate
			# x,y = self.random_position_in_world()
			# i_cell = self.coordinate_to_cell_index(x,y)
			# x_cell,y_cell = self.cell_index_to_cell_coordinate(i_cell)
			# ax.plot([x, x_cell],[y,y_cell],color='green')

			# test 3:
				# - pick a random cell 
				# - pick a random position in that cell 
				# - find the grid index 
				# - take it to cell index 
				# - plot coordinate of that cell 
			# i_x,i_y = self.cell_index_to_grid_index_map[0,:]
			# i_cell = self.grid_index_to_cell_index_map[i_x,i_y]	
			# x,y = self.random_position_in_cell(i_cell)
			# x_cell,y_cell = self.cell_index_to_cell_coordinate(self.coordinate_to_cell_index(x,y))
			# ax.plot([x, x_cell],[y,y_cell],color='green')

			# test 4: environment barrier???
			# x = -87.8
			# y = 41.81
			# x_safe,y_safe = self.environment_barrier((x,y))
			# ax.plot([x, x_safe],[y,y_safe],color='green')
