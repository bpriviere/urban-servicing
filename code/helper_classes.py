
import numpy as np 

class Gaussian:
	def __init__(self,i,x,y,th,s,v,nt):
		self.i = i
		self.s = s
		self.v = v 
		self.x = np.empty(nt)
		self.y = np.empty(nt)
		self.x[0] = x
		self.y[0] = y
		self.vx = np.cos(th)
		self.vy = np.sin(th)
		# self.change_dir()
		# print('self.x: ', self.x)
		# print('self.y: ', self.y)

	def move(self,p,timestep):
		self.x[timestep+1] = self.x[timestep] + p[0]
		self.y[timestep+1] = self.y[timestep] + p[1]

	def sample(self,timestep):
		x,y = np.random.normal([self.x[timestep],self.y[timestep]],self.s)
		return x,y

class CustomerModel:
	def __init__(self,param,env):
		self.param = param
		self.env = env

		# initialize customer model GMM
		nt = len(self.param.sim_times) 
		cgm_lst = []
		for i in range(self.param.cm_ng):
			
			if self.param.cm_linear_move:
				x0,y0 = [self.param.env_dx/2, 2*i*self.param.env_dy + self.param.env_dy/2]
				th0 = 0
			else:
				x0,y0 = self.env.random_position_in_world()
				th0 = np.random.random()*2*np.pi
			
			cgm_lst.append(
				Gaussian(i,x0,y0,th0,self.param.cm_sigma,self.param.cm_speed, nt))
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

			collision = True
			while collision:

				move = np.array([cgm.vx, cgm.vy])*cgm.v*dt
				cm_p_tp1 = [cgm.x[timestep] + move[0],cgm.y[timestep] + move[1]]
						
				collision = False
				if cm_p_tp1[0] > self.param.env_xlim[1] or cm_p_tp1[0] < self.param.env_xlim[0]:
					cgm.vx = -1*cgm.vx
					collision = True
				if cm_p_tp1[1] > self.param.env_ylim[1] or cm_p_tp1[1] < self.param.env_ylim[0]:
					cgm.vy = -1*cgm.vy
					collision = True

			cgm.move(move,timestep)

	def run_cm_model(self):
		for step,t in enumerate(self.param.sim_times[:-1]):
			self.move_cm(step)

	def eval_cm(self,timestep):
		# input: 
		# 	- self : env
		# 	- t : timestep of SIM
		# output: 
		# 	- cm : customer model probability matrix with shape: (env_nx,env_ny), where sum(sum(cm)) = 1 

		# for cgm in self.cgm_lst:
		# 	print('(cgm.x,cgm.y) = ({},{})'.format(cgm.x,cgm.y))

		cm = np.zeros((self.param.env_nx,self.param.env_ny))
		for i in range(self.param.cm_nsample_cm):
			x,y = self.sample_cm(timestep)
			x,y = self.env.environment_barrier([x,y])
			i_x,i_y = self.env.coordinate_to_grid_index(x,y)
			cm[i_x,i_y] += 1

		# normalize
		cm = cm/sum(sum(cm))
		return cm 

class Agent:
	def __init__(self,i,x,y,v,q,p):
		self.i = i
		self.x = x
		self.y = y
		self.v = v 
		self.q = q 
		self.mode = 0 # [dispatch, servicing]
		self.update = False
		self.p = p 

class Empty:
	def __init__(self):
		pass

class Dispatch:
	def __init__(self,x,y):
		self.x = x
		self.y = y 

class Service:
	def __init__(self,request):
		# [time_of_request,time_to_complete,x_p,y_p,x_d,y_d]
		self.time = request[0]
		self.time_to_complete = request[1]
		self.x_p = request[2]
		self.y_p = request[3]
		self.x_d = request[4]
		self.y_d = request[5]
		self.time_before_assignment = 0		
