
# standard package
# import matplotlib
import matplotlib.pyplot as plt 
import os, subprocess
import matplotlib.patches as patches
import numpy as np 
from matplotlib.backends.backend_pdf import PdfPages 

# my package
import utilities


# defaults
plt.rcParams.update({'font.size': 10})


class Plotter:
	def __init__(self,param):
		self.param = param 

	def save_figs(self,filename):
		fn = os.path.join( os.getcwd(), filename)
		pp = PdfPages(fn)
		for i in plt.get_fignums():
			pp.savefig(plt.figure(i))
			plt.close(plt.figure(i))
		pp.close()


	def open_figs(self,filename):
		pdf_path = os.path.join( os.getcwd(), filename)
		if os.path.exists(pdf_path):
			subprocess.call(["xdg-open", pdf_path])


	def make_fig(self,axlim = None):
		fig, ax = plt.subplots()
		if axlim is None:
			# ax.set_aspect('equal')
			pass
		else:
			ax.set_xlim(-axlim[0],axlim[0])
			ax.set_ylim(-axlim[1],axlim[1])
			ax.set_autoscalex_on(False)
			ax.set_autoscaley_on(False)
			ax.set_aspect('equal')
		return fig, ax


	def show_figs(self):
		plt.show()


	def plot_circle(self,x,y,r,fig=None,ax=None,title=None,label=None,color=None):
		if fig is None or ax is None:
			fig, ax = plt.subplots()

		circle = patches.Circle((x,y),radius=r)
			
		if color is not None:
			circle.set_color(color)
		if label is not None:
			circle.set_label(label)
		if title is not None:
			ax.set_title(title)

		ax.add_artist(circle)
		return circle 


	def plot_arrow(self,x,y,dx,dy,fig=None,ax=None,color=None,label=None):

		if fig is None or ax is None:
			fig, ax = plt.subplots()

		scale = param.plot_arrow_length/np.linalg.norm([dx,dy])
		dx *= scale
		dy *= scale

		line = ax.arrow(x,y,dx,dy,width=param.plot_arrow_width,head_width=param.plot_arrow_head_width,head_length=param.plot_arrow_head_length)

		if color is not None:
			line.set_color(color)
		if label is not None:
			line.set_label(label)
		return line 


	def plot_line(self,x1,y1,x2,y2,fig=None,ax=None,color=None,label=None): 

		if fig is None or ax is None:
			fig, ax = plt.subplots()

		line = ax.plot([x1,x2],[y1,y2],ls='--')[0]

		if color is not None:
			line.set_color(color)
		if label is not None:
			line.set_label(label)
		return line	

	def plot_dashed(self,x,y,r,fig=None,ax=None,title=None,label=None,color=None):
		if fig is None or ax is None:
			fig, ax = plt.subplots()

		circle = patches.Circle((x,y),radius=r,fill=False,ls='--')
			
		if color is not None:
			circle.set_color(color)
		if label is not None:
			circle.set_label(label)
		if title is not None:
			ax.set_title(title)

		ax.add_artist(circle)
		return circle 	


	def plot_rectangle(self,x,y,r,fig=None,ax=None,title=None,label=None,color=None,angle=None):
		if fig is None or ax is None:
			fig, ax = plt.subplots()

		if angle is not None:
			rect = patches.Rectangle((x,y),height=r,width=r,angle=angle)
		else:
			shift = r/2
			x -= shift
			y -= shift
			rect = patches.Rectangle((x,y),height=r,width=r)
			
		if color is not None:
			rect.set_color(color)
		if label is not None:
			rect.set_label(label)
		if title is not None:
			ax.set_title(title)

		ax.add_artist(rect)
		return rect 

	def plot_gridworld_dataset(self,env):

		# for each cell, plot random generation of customer demand and eta over time
		
		dataset = env.dataset

		fig_eta,ax_eta = plt.subplots()
		fig_c,ax_c = plt.subplots()
		for i_x in range(param.env_nx):
			for i_y in range(param.env_ny):

				eta_ij = []
				c_ij = [] 
				for step,time in enumerate(param.sim_times):
					eta_ij.append(env.eta_cell(i_x,i_y,time))
					c_ij.append(env.customer_distribution_matrix(time)[i_x,i_y])

				ax_c.plot(param.sim_times,c_ij,label='(x,y) = {}, {}'.format(i_x,i_y))
				ax_eta.plot(param.sim_times,eta_ij,label='(x,y) = {}, {}'.format(i_x,i_y))
		
		ax_c.legend()
		ax_eta.legend()
		ax_c.set_title('Customer Demand')
		ax_eta.set_title('ETA')
				
	def plot_sim_rewards(self,sim_results):

		fig1,ax1 = self.make_fig()
		fig2,ax2 = self.make_fig()
		for controller_name, sim_result in sim_results.items():
			ax1.plot(sim_result["times"], sim_result["rewards"], label=controller_name)
			ax2.plot(sim_result["times"], np.cumsum(sim_result["rewards"]), label=controller_name)
		ax1.legend()
		ax2.legend()
		ax1.set_title('Reward at each timestep')
		ax2.set_title('Cumulative Reward')

	def sim_plot(self,controller_name,results,timestep):
		# input: 
		# 	- results is a dictionary of results for some controller
		# 	- timestep is when to plot

		fig = plt.figure()
		axs = []
		ncol = 3
		nrow = int(np.floor(len(self.param.state_keys)/ncol)) + 1
		for i_key, key in enumerate(self.param.state_keys):
			num = i_key + 1
			curr_ax = fig.add_subplot(nrow,ncol,num)
			curr_ax.set_title(key)
			
			if 'action' in key:
				im_to_plot = results["agents_ave_vec_action_distribution"][timestep]
				self.ave_actions_vector_plot(im_to_plot,xlim=self.param.env_xlim,ylim=self.param.env_ylim,fig=fig,ax=curr_ax)

			elif 'distribution' in key:
				# im coordinates -> try to change this one to sim coordinates
				if 'value' in key:
					agent_idx = 0
					im_to_plot = results[key][timestep][agent_idx]
				else:
					im_to_plot = results[key][timestep]

				im = curr_ax.imshow(self.sim_to_im_coordinate(im_to_plot),
					vmin=0,vmax=1,cmap='gray_r',
					extent=[self.param.env_xlim[0],self.param.env_xlim[1],self.param.env_ylim[0],self.param.env_ylim[1]])

			elif 'location' in key:
				locs = results[key][timestep] # sim coordinates ... 
				if np.size(locs) > 0:
					curr_ax.scatter(locs[:,0],locs[:,1])

			elif 'operation' in key:
				nmode = 2
				hist = np.zeros((nmode))
				for i_mode in range(nmode):
					hist[i_mode] = sum(results[key][timestep] == i_mode) / self.param.ni
				curr_ax.bar(range(nmode),hist)
				curr_ax.set_ylim([0,1])
				plt.xticks(range(nmode),self.param.mode_names,rotation='vertical')
			
			if 'distribution' in key or 'location' in key or 'action' in key:
				# create same axis
				curr_ax.set_xticks(self.param.env_x) 
				curr_ax.set_yticks(self.param.env_y) 
				curr_ax.set_xlim(self.param.env_xlim)
				curr_ax.set_ylim(self.param.env_ylim)
				curr_ax.set_aspect('equal')
				curr_ax.grid(True)

			axs.append(curr_ax)

		# # colorbar
		# fig.subplots_adjust(right=0.8)
		# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		# fig.colorbar(im, cax=cbar_ax)

		# title
		t = self.param.sim_times[timestep]
		T = self.param.sim_times[-1]
		fig.suptitle('{} at t/T = {}/{}'.format(controller_name,t,T))

	def plot_distribution_over_time(self,results,env):
		for timestep,time in enumerate(results.times):
			self.plot_distribution(results, timestep, env)

	def sim_plot_over_time(self,controller_name,sim_result):
		for timestep,time in enumerate(sim_result["times"]):
			self.sim_plot(controller_name, sim_result, timestep)	

	def sim_to_im_coordinate(self,im):
		# im coordinates
		im = im.T
		im = np.flipud(im)
		return im 

	def ave_actions_vector_plot(self,im_ave_vec_action,xlim=None,ylim=None,fig=None,ax=None):
		# plots average action at each state as a vector
		# input:
		# 	- im_vec_action (nx,ny,2)
		# 	- plot param

		dx = self.param.env_dx
		X,Y = np.meshgrid(self.param.env_x + self.param.env_dx/2,self.param.env_y + self.param.env_dy/2)
		U = np.zeros(X.shape)
		V = np.zeros(X.shape)
		C = np.zeros(X.shape)

		for i_x,_ in enumerate(self.param.env_x):
			for i_y,_ in enumerate(self.param.env_y):
				U[i_y,i_x] = im_ave_vec_action[i_x,i_y,0]
				V[i_y,i_x] = im_ave_vec_action[i_x,i_y,1]
				C[i_y,i_x] = np.linalg.norm(im_ave_vec_action[i_x,i_y,:])

		# normalize arrow length
		idx = np.nonzero(U**2 + V**2)
		U[idx] = U[idx] / np.sqrt(U[idx]**2 + V[idx]**2);
		V[idx] = V[idx] / np.sqrt(U[idx]**2 + V[idx]**2);

		im = ax.quiver(X[idx],Y[idx],U[idx],V[idx],C[idx]) #,scale_units='xy')

	def macro_sim_plot(self,macro_sim_results,ni_lst):

		# macro_sim_results is a list of sim_results, for each macro_sim parameter
		# 	sim_results is a dictionary of [key,value] = [controller_name,sim_result]
		# 		sim_result is a dictionary of [key,value] output by sim.run.py

		# temp
		macro_sim_dict = dict()
		macro_sim_dict["ni"] = ni_lst

		macro_key = 'ni'
		macro_values = macro_sim_dict["ni"]


		# use some macro param 
		macro_param = macro_sim_results[0]["param"]
		controller_names = macro_param["controllers"]

		# assign markers by controller
		marker_dict = dict()
		color_dict = dict()
		marker_count = 0
		for controller_name in controller_names:
			marker_dict[controller_name] = macro_param["plot_markers"][marker_count]
			color_dict[controller_name] = macro_param["plot_colors"][marker_count]
			marker_count += 1 

		rewards_controller_macro_values = []
		for k_value,macro_value in enumerate(macro_values):
			sim_result = macro_sim_results[k_value]
			rewards_controller_macro_value = dict()
			for controller_name in controller_names:
				rewards_controller_macro_value[controller_name] = sim_result[controller_name]["rewards"]
			rewards_controller_macro_values.append(rewards_controller_macro_value)

		fig1,ax1 = self.make_fig()
		for i,rewards_controller_macro_value in enumerate(rewards_controller_macro_values):
			for controller_name in controller_names:
				reward = rewards_controller_macro_value[controller_name]
				
				if i>0:
					ax1.plot(macro_sim_dict[macro_key][i], sum(reward), 
						marker=marker_dict[controller_name],
						color=color_dict[controller_name])
				else:
					ax1.plot(macro_sim_dict[macro_key][i], sum(reward), 
						marker=marker_dict[controller_name],
						color=color_dict[controller_name],
						label=controller_name)

		ax1.legend()



