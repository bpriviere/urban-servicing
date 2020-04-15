
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

		# make polygon with union shape
		print('   merge polygons in mapfile...')
		union = unary_union(shapes)
		if union.geom_type == 'MultiPolygon':
			self.city_polygon = union[0]			
			for polygon in union:
				if polygon.area > self.city_polygon.area:
					self.city_polygon = polygon
		else:
			self.city_polygon = union

		# make negative polygon image 


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
		print('   making geometry mask...')
		valid_geometery_cell_mask = self.get_geometry_mask()

		occupancy_grid_mask_on = False
		if occupancy_grid_mask_on:
			occupancy_cell_mask = self.get_occupancy_mask()
			full_mask = np.logical_and(valid_geometery_cell_mask, occupancy_cell_mask)
		else:
			full_mask = valid_geometery_cell_mask

		# make utility maps 
		# 	- grid_index_to_cell_index_map
		# 	- cell_index_to_grid_index_map
		print('   reducing map...')
		self.reduce_map(full_mask)

		# 
		# print('   refining map...')
		# todo 
		xmin = np.inf
		xmax = -np.inf
		ymin = np.inf
		ymax = -np.inf
		for i_x,x in enumerate(self.param.env_x):
			for i_y,y in enumerate(self.param.env_y):
				if full_mask[i_x,i_y]:
					if self.param.env_x[i_x] > xmax:
						xmax = self.param.env_x[i_x]
					if self.param.env_x[i_x] < xmin:
						xmin = self.param.env_x[i_x]
					if self.param.env_y[i_y] > ymax:
						ymax = self.param.env_y[i_y]
					if self.param.env_y[i_y] < ymin:
						ymin = self.param.env_y[i_y]			
		self.valid_xmin = xmin
		self.valid_xmax = xmax
		self.valid_ymin = ymin
		self.valid_ymax = ymax

		# kind of awkward 
		self.param.env_ncell = self.cell_index_to_grid_index_map.shape[0]
		self.param.nq = self.param.env_ncell*self.param.env_naction


	def get_geometry_mask(self):
		# label valid cells where valid cells are either (inclusive OR)
		# 	- center is in the self.city_polygon  
		# 	- contain an element of the boundary 

		valid_cells_mask = np.zeros((len(self.param.env_x),len(self.param.env_y)),dtype=bool)
		
		xmin_thresh = -87.8
		xmax_thresh = -87.575
		
		ymin_thresh = 41.8 
		ymax_thresh = 42.0 

		# x,y are bottom left hand corner of cell 
		for i_x,x in enumerate(self.param.env_x):
			for i_y,y in enumerate(self.param.env_y):
				if y > ymin_thresh and x > xmin_thresh and x < xmax_thresh and y < ymax_thresh:
					cell_center = Point((x + self.param.env_dx/2, y + self.param.env_dy/2))
					if cell_center.within(self.city_polygon): 
						valid_cells_mask[i_x,i_y] = True

					else:
						for point in self.city_boundary:
							if self.point_in_cell(point,x,y):
								valid_cells_mask[i_x,i_y] = True
								break # out of points-in-city-boundary loop
		return valid_cells_mask

	def get_occupancy_mask(self):
		
		occupancy_cells_mask = np.zeros((len(self.param.env_x),len(self.param.env_y)),dtype=bool)

		for customer in np.vstack((self.train_dataset,self.test_dataset)):
			# [time_of_request,time_to_complete,x_p,y_p,x_d,y_d]
			i_x,i_y = self.coordinate_to_grid_index(customer[2],customer[3])
			occupancy_cells_mask[i_x,i_y] = True 
			try:
				i_x,i_y = self.coordinate_to_grid_index(customer[3],customer[4])
				occupancy_cells_mask[i_x,i_y] = True 
			except:
				pass 

		return occupancy_cells_mask

	def refine_map(self):
		# this splits the remaining cells into the desired number of cells 
		pass 		

	def reduce_map(self,mask):
		ncells = sum(sum(mask))
		self.grid_index_to_cell_index_map = np.zeros((mask.shape),dtype=int)
		self.cell_index_to_grid_index_map = np.zeros((ncells,2),dtype=int)
		count = 0
		for i_x,x in enumerate(self.param.env_x):
			for i_y,y in enumerate(self.param.env_y):		
				if mask[i_x,i_y]:
					self.grid_index_to_cell_index_map[i_x,i_y] = count
					self.cell_index_to_grid_index_map[count,:] = [i_x,i_y]
					count += 1

	def print_map(self,fig=None,ax=None):
		ax.plot(self.city_boundary[:,0],self.city_boundary[:,1],linewidth=1)
		ax.set_xticks(self.param.env_x)
		ax.set_yticks(self.param.env_y)
		ax.set_xlim([self.param.env_xlim[0],self.param.env_xlim[1]])
		ax.set_ylim([self.param.env_ylim[0],self.param.env_ylim[1]])

		for i in range(self.param.env_ncell):
			x,y = self.cell_index_to_cell_coordinate(i)
			x += self.param.env_dx/2
			y += self.param.env_dx/2
			ax.scatter(x,y)

		ax.axhline(self.valid_ymax)
		ax.axhline(self.valid_ymin)
		ax.axvline(self.valid_xmax)
		ax.axvline(self.valid_xmin)

		ax.grid(True)



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

	def heatmap(self, dataset):

		customer_demand_per_state = np.zeros((self.param.env_ncell))
		n_customers = dataset.shape[0]

		# print('self.city_polygon.bounds:',self.city_polygon.bounds)
		# print('self.param.env_x: ',self.param.env_x)
		# print('self.param.env_y: ',self.param.env_y)

		for customer in dataset:
			# [time_of_request,time_to_complete,x_p,y_p,x_d,y_d]
			# print('customer[2]: ', customer[2])
			# print('customer[3]: ', customer[3])

			intense_on = False
			if intense_on:
				customer_pickup = Point((customer[2], customer[3]))
				if customer_pickup.within(self.city_polygon):
					s = self.coordinate_to_cell_index(customer[2],customer[3])
					customer_demand_per_state[s] += 1
			else:
				s = self.coordinate_to_cell_index(customer[2],customer[3])
				customer_demand_per_state[s] += 1

		customer_demand_per_state /= n_customers

		# convert to im
		im_customer_demand = np.zeros((self.param.env_nx,self.param.env_ny))
		for i in range(self.param.env_ncell):
			# print('self.cell_index_to_grid_index_map[i]:',self.cell_index_to_grid_index_map[i])
			i_x,i_y = self.cell_index_to_grid_index_map[i]
			im_customer_demand[i_x,i_y] = customer_demand_per_state[i]

		return im_customer_demand 



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
