

import numpy as np 

import utilities
from param import Param


param = Param()

class Agent:
	def __init__(self,i,x,y,v,q):
		self.i = i
		self.x = x
		self.y = y
		self.v = v 
		self.q = q 
		self.mode = 0 # [idle, servicing, pickup]
		self.update = False
		self.p = param.initial_covariance
		
	# def step(self,action,env):

	# 	wait_time = 0 
	# 	if isinstance(action,Service):
	# 		self.service = action
	# 		time = env.param.sim_times[env.timestep] 
	# 		eta = env.eta(self.x,self.y,self.service.x_p,self.service.y_p) 
	# 		wait_time = eta + self.service.time_before_assignment
	# 		self.update = True

	# 		# initialize service
	# 		self.mode = 1
	# 		self.pickup_vector = np.array([self.service.x_p - self.x, self.service.y_p-self.y])
	# 		self.pickup_dist = np.linalg.norm(self.pickup_vector)
	# 		self.pickup_vector = self.pickup_vector / self.pickup_dist
	# 		self.pickup_speed = self.pickup_dist/eta
	# 		self.pickup_finish = time + eta

	# 		self.dropoff_vector = np.array([self.service.x_d - self.service.x_p,self.service.y_d-self.service.y_p])
	# 		self.dropoff_dist = np.linalg.norm(self.dropoff_vector)
	# 		self.dropoff_vector = self.dropoff_vector/self.dropoff_dist
	# 		self.dropoff_speed = self.dropoff_dist/self.service.time_to_complete
	# 		self.dropoff_finish = self.pickup_finish + self.service.time_to_complete

	# 		# print('time: ',time)
	# 		# print('eta: ',eta)
	# 		# print('pickup_finish: ',self.pickup_finish)
	# 		# exit()

	# 		# start moving
	# 		next_time = env.param.sim_times[env.timestep+1]
	# 		if next_time > self.pickup_finish:
	# 			self.x = self.service.x_p
	# 			self.y = self.service.y_p 
	# 			self.mode = 2
	# 		else:
	# 			dt = next_time-time
	# 			self.x = self.x + self.pickup_vector[0]*self.pickup_speed*dt
	# 			self.y = self.y + self.pickup_vector[1]*self.pickup_speed*dt

	# 	elif isinstance(action,Dispatch):

	# 		self.dispatch = action 
	# 		self.mode = 3

	# 		# print('Dispatch')
	# 		# print('agent {}: (x,y) = ({},{})'.format(self.i,self.x,self.y))
	# 		# print('(dispatch.x,dispatch.y) = ({},{}): '.format(action.x,action.y))
	# 		# exit()

	# 		# assign dispatch move 
	# 		time = env.param.sim_times[env.timestep] 
	# 		eta = env.eta(self.x,self.y,self.dispatch.x,self.dispatch.y)
	# 		self.move_vector = np.array([self.dispatch.x - self.x, self.dispatch.y-self.y])
	# 		self.move_dist = np.linalg.norm(self.move_vector)
	# 		self.move_vector = self.move_vector/self.move_dist
	# 		self.move_speed = self.move_dist/eta 
	# 		self.move_finish = time+eta 

	# 		# start moving 
	# 		next_time = env.param.sim_times[env.timestep+1]
	# 		if next_time > self.move_finish:
	# 			self.x = self.dispatch.x
	# 			self.y = self.dispatch.y
	# 			self.mode = 0
	# 		else:
	# 			dt = next_time-time
	# 			self.x = self.x + self.move_vector[0]*self.move_speed*dt
	# 			self.y = self.y + self.move_vector[1]*self.move_speed*dt
							
	# 	elif isinstance(action,Empty):
			
	# 		next_time = env.param.sim_times[env.timestep+1]
	# 		time = env.param.sim_times[env.timestep]
	# 		dt = next_time - time 

	# 		# pickup mode 
	# 		if self.mode == 1:
	# 			if next_time > self.pickup_finish:
	# 				self.x = self.service.x_p
	# 				self.y = self.service.y_p 
	# 				self.mode = 2
	# 			else:
	# 				self.x = self.x + self.pickup_vector[0]*self.pickup_speed*dt
	# 				self.y = self.y + self.pickup_vector[1]*self.pickup_speed*dt

	# 		# dropoff mode 
	# 		elif self.mode == 2:
	# 			if next_time > self.dropoff_finish:
	# 				self.x = self.service.x_d
	# 				self.y = self.service.y_d 
	# 				self.mode = 0
	# 			else:
	# 				self.x = self.x + self.dropoff_vector[0]*self.dropoff_speed*dt
	# 				self.y = self.y + self.dropoff_vector[1]*self.dropoff_speed*dt

	# 		# dispatch mode 
	# 		elif self.mode == 3:
	# 			if next_time > self.move_finish:
	# 				self.x = self.dispatch.x
	# 				self.y = self.dispatch.y
	# 				self.mode = 0
	# 			else:
	# 				dt = next_time-time
	# 				self.x = self.x + self.move_vector[0]*self.move_speed*dt
	# 				self.y = self.y + self.move_vector[1]*self.move_speed*dt

	# 	# make sure you don't leave environment
	# 	self.x,self.y = utilities.environment_barrier([self.x,self.y])

	# 	return wait_time 

# class Empty:
# 	def __init__(self):
# 		pass

# class Dispatch:
# 	def __init__(self,x,y):
# 		self.x = x
# 		self.y = y 

# class Service:
# 	def __init__(self,request):
# 		# [time_of_request,time_to_complete,x_p,y_p,x_d,y_d]
# 		self.time = request[0]
# 		self.time_to_complete = request[1]
# 		self.x_p = request[2]
# 		self.y_p = request[3]
# 		self.x_d = request[4]
# 		self.y_d = request[5]
# 		self.time_before_assignment = 0

