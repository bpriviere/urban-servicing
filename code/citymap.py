
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

		# make boundary
		x,y = self.city_polygon.exterior.coords.xy
		self.city_boundary = np.asarray([x,y]).T # [npoints x 2]

		# make grid 
		self.param.env_xlim = [self.param.xmin_thresh,self.param.xmax_thresh]
		self.param.env_ylim = [self.param.ymin_thresh,self.param.ymax_thresh]

		self.param.update()

		self.grid_index_to_cell_index_map = np.zeros((self.param.env_nx,self.param.env_ny),dtype=int)
		self.cell_index_to_grid_index_map = np.zeros((self.param.env_ncell,2),dtype=int)
		count = 0
		for i_y,y in enumerate(self.param.env_y):
			for i_x,x in enumerate(self.param.env_x):
				self.grid_index_to_cell_index_map[i_x,i_y] = count
				self.cell_index_to_grid_index_map[count,:] = [i_x,i_y]
				count += 1

		print('   making geometry mask...')
		self.valid_cells = self.get_valid_cells_list()

		self.valid_xmin = self.param.env_x[0]
		self.valid_xmax = self.param.env_x[-1] + self.param.env_dx
		self.valid_ymin = self.param.env_y[0]
		self.valid_ymax = self.param.env_y[-1] + self.param.env_dy		

		take_invalid_cells = True
		if take_invalid_cells:
			# scales the ns, na , nq for valid and invalid cells, but more clean 
			self.param.env_ncell = count 
		else:
			# removes the invalid cells
			self.param.env_ncell = len(self.valid_cells) 

		self.param.nq = self.param.env_ncell*self.param.env_naction
		print('self.param.env_ncell:',self.param.env_ncell)


	def get_valid_cells_list(self):
		# label valid cells where valid cells are either (inclusive OR)
		# 	- center is in the self.city_polygon  
		# 	- contain an element of the boundary 

		valid_cells_lst = [] 
		
		# x,y are bottom left hand corner of cell 
		for i_x,x in enumerate(self.param.env_x):
			for i_y,y in enumerate(self.param.env_y):

				temp = False
				if temp:
					valid_cells_lst.append( self.grid_index_to_cell_index_map[i_x,i_y])

				else:
					cell_center = Point((x + self.param.env_dx/2, y + self.param.env_dy/2))
					if cell_center.within(self.city_polygon): 
						valid_cells_lst.append( self.grid_index_to_cell_index_map[i_x,i_y])

					# else:
					# 	for point in self.city_boundary:
					# 		if self.point_in_cell(point,x,y):
					# 			valid_cells_lst.append( self.grid_index_to_cell_index_map[i_x,i_y])
					# 			break # out of points-in-city-boundary loop
		return valid_cells_lst



	def print_map(self,fig=None,ax=None):
		ax.plot(self.city_boundary[:,0],self.city_boundary[:,1],linewidth=1)
		ax.set_xticks(self.param.env_x)
		
		import matplotlib.pyplot as plt 
		plt.setp( ax.xaxis.get_majorticklabels(), rotation=90, fontsize = 5 )
		
		ax.set_yticks(self.param.env_y)
		# ax.set_xlim([self.param.env_xlim[0],self.param.env_xlim[1]])
		# ax.set_ylim([self.param.env_ylim[0],self.param.env_ylim[1]])
		eps = 1e-3
		ax.set_xlim([self.city_polygon.bounds[0]-eps,self.city_polygon.bounds[2]+eps])
		ax.set_ylim([self.city_polygon.bounds[1]-eps,self.city_polygon.bounds[3]+eps])

		for i in range(self.param.env_ncell):
			if i in self.valid_cells:
				x,y = self.cell_index_to_cell_coordinate(i)
				x += self.param.env_dx/2
				y += self.param.env_dy/2
				ax.scatter(x,y)

		ax.axhline(self.valid_ymax)
		ax.axhline(self.valid_ymin)
		ax.axvline(self.valid_xmax)
		ax.axvline(self.valid_xmin)

		ax.grid(True)

		print('self.param.env_x:',self.param.env_x)
		print('self.param.env_y:',self.param.env_y)



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
		return (point[0] >= x and point[0] <= x + self.param.env_dx) and \
			(point[1] >= y and point[1] <= y + self.param.env_dy)

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