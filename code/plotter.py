
# standard package
import matplotlib.pyplot as plt 
import os, subprocess
import matplotlib.patches as patches
import numpy as np 
from matplotlib.backends.backend_pdf import PdfPages 

# my package
from param import Param 
import utilities

param = Param()

def save_figs(filename):
	fn = os.path.join( os.getcwd(), filename)
	pp = PdfPages(fn)
	for i in plt.get_fignums():
		pp.savefig(plt.figure(i))
		plt.close(plt.figure(i))
	pp.close()


def open_figs(filename):
	pdf_path = os.path.join( os.getcwd(), filename)
	if os.path.exists(pdf_path):
		subprocess.call(["xdg-open", pdf_path])


def make_fig(axlim = None):
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


def show_figs():
	plt.show()


def plot_circle(x,y,r,fig=None,ax=None,title=None,label=None,color=None):
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


def plot_arrow(x,y,dx,dy,fig=None,ax=None,color=None,label=None):

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


def plot_line(x1,y1,x2,y2,fig=None,ax=None,color=None,label=None): 

	if fig is None or ax is None:
		fig, ax = plt.subplots()

	line = ax.plot([x1,x2],[y1,y2],ls='--')[0]

	if color is not None:
		line.set_color(color)
	if label is not None:
		line.set_label(label)
	return line	

def plot_dashed(x,y,r,fig=None,ax=None,title=None,label=None,color=None):
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


def plot_rectangle(x,y,r,fig=None,ax=None,title=None,label=None,color=None,angle=None):
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

def plot_gridworld_dataset(env):

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


			
def plot_sim_rewards(sim_results):

	fig1,ax1 = make_fig()
	fig2,ax2 = make_fig()
	for sim_result in sim_results:
		ax1.plot(sim_result.times, sim_result.rewards, label=sim_result.name)
		ax2.plot(sim_result.times, np.cumsum(sim_result.rewards), label=sim_result.name)
	ax1.legend()
	ax2.legend()
	ax1.set_title('Reward at each timestep')
	ax2.set_title('Cumulative Reward')

def plot_sim_results(sim_results, key):

	for sim_result in sim_results: # for each controller
		fig1,ax1 = make_fig()
		values_agents = getattr(sim_result, key) # [nt, ni, dim_key]
		values_agents = np.swapaxes(values_agents,0,2) # [dim_key, ni, nt]
		for k_val, value_agents in enumerate(values_agents):
			for k_agent,value_agent in enumerate(value_agents):
				if k_agent == 0:
					ax1.plot(sim_result.times, value_agent,label='{}[{}]'.format(key,k_val))
				else:
					ax1.plot(sim_result.times, value_agent)
				
				if sim_result.name == "ctd":
					break

		ax1.set_title("{}: {}".format(sim_result.name,key))
		ax1.legend()

def plot_bellman_q(env):

	q = np.zeros((param.nq, param.sim_nt))
	dataset = env.dataset
	for step,time in enumerate(param.sim_times):
		dataset_t = dataset[dataset[:,0]<=time,:]
		_,q[:,step] = utilities.solve_MDP(env,dataset_t,time)

	fig1,ax1 = make_fig()
	for i,q_i in enumerate(q):
		ax1.plot(param.sim_times, q_i, label = 'q_value[{}]'.format(i))
	ax1.set_title('Bellman Q')
	ax1.legend()

def plot_distribution_error(env):

	# customer distribution
	im_customer = env.eval_cm(env.timestep)
	
	# agent distribution 
	im_agent = np.zeros((env.param.env_nx,env.param.env_ny))

	# error
	im_err = np.abs(im_agent-im_customer)

	# align axis with image coordinate system 
	im_err = im_err.T
	im_err = np.flipud(im_err)

	# plot
	fig,ax = make_fig()
	ax.set_title('Distribution Error')
	ax.set_xticks(env.param.env_x)
	ax.set_yticks(env.param.env_y)
	ax.set_xlim(env.param.env_xlim)
	ax.set_ylim(env.param.env_ylim)
	ax.set_aspect('equal')
	ax.grid(True)
	axim=ax.imshow(im_err,cmap='gray_r',vmin=0,vmax=1, 
		extent=[env.param.env_xlim[0],env.param.env_xlim[1],env.param.env_ylim[0],env.param.env_ylim[1]])
	fig.colorbar(axim)

def plot_value_fnc(env):

	v = utilities.q_value_to_value_fnc(env,env.agents[0].q)
	im_v = np.zeros((env.param.env_nx,env.param.env_ny))
	for i in range(env.param.env_ncell):
		i_x,i_y = utilities.cell_index_to_xy_cell_index(i)
		im_v[i_x,i_y] = v[i]

	im_v = im_v.T
	im_v = np.flipud(im_v)

	fig,ax = make_fig()
	ax.set_title('Value Function')
	ax.set_xticks(env.param.env_x)
	ax.set_yticks(env.param.env_y)
	ax.set_xlim(env.param.env_xlim)
	ax.set_ylim(env.param.env_ylim)
	ax.set_aspect('equal')
	ax.grid(True)
	axim=ax.imshow(im_v,
		extent=[env.param.env_xlim[0],env.param.env_xlim[1],env.param.env_ylim[0],env.param.env_ylim[1]])
	fig.colorbar(axim)


def plot_distribution(results, timestep, env):

	# makes a figure with 3 distribution subplots at a single timestep
	# inputs: 
	# 	- results: named tuple: ['name','times','rewards','agent_operation','agent_locations','agent_q_values'] (todo.. change to numpy array)
	# 	- timestep: int
	# 	- env: todo... eliminate this input somehow 
	# outputs: 
	# 	fig, (ax1,ax2,...)	

	# some note 
	# passing the env makes it difficult to just run the plots with only the data

	# make figs, 1x3 subplots, wide figure
	fig, (ax1,ax2,ax3) = plt.subplots(1, 3, sharey=True)
	title = '{} at t/T = {}/{}'.format(results.name, results.times[timestep],results.times[-1])
	fig.suptitle(title)

	# ax1.set_xticks(env.param.env_x)
	# ax1.set_yticks(env.param.env_y)
	# ax1.set_xlim(env.param.env_xlim)
	# ax1.set_ylim(env.param.env_ylim)
	# ax2.set_xticks(env.param.env_x)
	# ax2.set_yticks(env.param.env_y)
	# ax2.set_xlim(env.param.env_xlim)
	# ax2.set_ylim(env.param.env_ylim)
	# ax3.set_xticks(env.param.env_x)
	# ax3.set_yticks(env.param.env_y)
	# ax3.set_xlim(env.param.env_xlim)
	# ax3.set_ylim(env.param.env_ylim)
	# ax1.grid(True)
	# ax2.grid(True)
	# ax3.grid(True)

	# get images
	im_gmm = env.get_im_gmm(timestep)
	im_v = env.get_im_v(timestep,results)
	im_agent = env.get_im_agent(timestep,results)
	# im_agent = env.get_im_free_agent(timestep,results)

	# print('im_gmm',im_gmm)
	# print('im_v',im_v)
	# print('im_agent',im_agent)

	# customer plot 
	# plot evaluation of gaussian mixture model
	ax1.set_title('Gaussian Mixture Model')
	# ax1.imshow(im_gmm)
	ax1.imshow(im_gmm,vmin=0,vmax=1)
	# ax1.imshow(im_gmm, 
	# 	extent=[env.param.env_xlim[0],env.param.env_xlim[1],env.param.env_ylim[0],env.param.env_ylim[1]])

	# todo... later
	# plot actual dataset 
	# ax?.set_title('Customer Distribution')
	# ax?.imshow(im_cd, 
	# 	extent=[env.param.env_xlim[0],env.param.env_xlim[1],env.param.env_ylim[0],env.param.env_ylim[1]])
	
	# value fnc plot (for a single agent?)
	ax2.set_title('Value Function')
	# im2 = ax2.imshow(im_v)
	im2 = ax2.imshow(im_v,vmin=0,vmax=1)
	# im2 = ax2.imshow(im_v,
	# 	extent=[env.param.env_xlim[0],env.param.env_xlim[1],env.param.env_ylim[0],env.param.env_ylim[1]])

	# agent dist plot 
	ax3.set_title('Agent Distribution')
	# im3 = ax3.imshow(im_agent)
	im3 = ax3.imshow(im_agent,vmin=0,vmax=1)
	# im3 = ax3.imshow(im_agent, 
	# 	extent=[env.param.env_xlim[0],env.param.env_xlim[1],env.param.env_ylim[0],env.param.env_ylim[1]])

	# colorbar
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(im3, cax=cbar_ax)


def plot_distribution_over_time(results,env):

	for timestep,time in enumerate(results.times):
		plot_distribution(results, timestep, env)