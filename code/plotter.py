
# standard package
# import matplotlib
import matplotlib.pyplot as plt 
import os, subprocess
import matplotlib.patches as patches
import numpy as np 
from matplotlib.backends.backend_pdf import PdfPages 
import matplotlib

# defaults
plt.rcParams.update({'font.size': 10})
plt.rcParams['lines.linewidth'] = 4


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
	
	dataset = env.test_dataset

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
	for controller_name, sim_result in sim_results.items():
		ax1.plot(sim_result["times"], sim_result["rewards"], label=controller_name)
		ax2.plot(sim_result["times"], np.cumsum(sim_result["rewards"]), label=controller_name)
	ax1.legend()
	ax2.legend()
	ax1.set_title('Reward at each timestep')
	ax2.set_title('Cumulative Reward')

def sim_plot(controller_name,results,timestep,fig,city_boundary=None,agent_colors=None):
	# input: 
	# 	- results is a dictionary of results for some controller
	# 	- timestep is when to plot

	param = results["param"]
	state_keys = param["state_keys"]
	plot_keys = param["plot_keys"]

	axs = []
	ncol = 3
	nrow = int(np.floor(len(state_keys)/ncol)) + 1
	for i_key, key in enumerate(plot_keys):
		num = i_key + 1
		curr_ax = fig.add_subplot(nrow,ncol,num)
		# curr_ax.set_title(key,fontsize=16)
		if 'action' in key:
			im_to_plot = results["agents_ave_vec_action_distribution"][timestep]
			ave_actions_vector_plot(param,im_to_plot,xlim=param["env_xlim"],ylim=param["env_ylim"],fig=fig,ax=curr_ax)

		elif 'distribution' in key:
			# im coordinates -> try to change this one to sim coordinates
			if 'value' in key:
				agent_idx = 0
				im_to_plot = results[key][timestep][agent_idx]
			else:
				im_to_plot = results[key][timestep]

			im = curr_ax.imshow(sim_to_im_coordinate(im_to_plot),
				vmin=0,vmax=1,cmap='gray_r',
				extent=[param["env_xlim"][0],param["env_xlim"][1],param["env_ylim"][0],param["env_ylim"][1]])

			# im = curr_ax.imshow(sim_to_im_coordinate(im_to_plot),vmin=0,vmax=1,cmap='gray_r')

		elif 'location' in key:

			# service_color = 'orange'
			# dispatch_color = 'blue'
			# locs = results[key][timestep] # sim coordinates ... 
			# if np.size(locs) > 0:
			# 	if 'agent' in key:
			# 		service_agent_idx = np.asarray(results["agents_operation"][timestep],dtype=bool)
			# 		curr_ax.scatter(locs[service_agent_idx,0],locs[service_agent_idx,1],c=service_color)
			# 		curr_ax.scatter(locs[~service_agent_idx,0],locs[~service_agent_idx,1],c=dispatch_color)

			# 	elif 'customer' in key:
			# 		curr_ax.scatter(locs[:,0],locs[:,1],c=service_color)

			# agent_colors is a lst of lst: first index tells service/dispatch and second index tells which agent
			service_color = 'orange'
			dispatch_color = 'blue'
			locs = results[key][timestep] # sim coordinates ... 
			if np.size(locs) > 0:
				if 'agent' in key:
					if agent_colors is None:
						service_agent_idx = np.asarray(results["agents_operation"][timestep],dtype=bool)
						curr_ax.scatter(locs[service_agent_idx,0],locs[service_agent_idx,1],c=service_color)
						curr_ax.scatter(locs[~service_agent_idx,0],locs[~service_agent_idx,1],c=dispatch_color)
					else:

						# alphas = np.linspace(0.5, 1, param["ni"])
						# rgba_red = np.zeros((10,4))
						# rgba_red[:,0] = 1.0
						# rgba_red[:, 3] = alphas
						# rgba_blue = np.zeros((10,4))
						# rgba_blue[:, 3] = alphas
						# curr_ax.scatter(locs[service_agent_idx,0],locs[service_agent_idx,1],color=rgba_red[service_agent_idx])
						# curr_ax.scatter(locs[service_agent_idx,0],locs[service_agent_idx,1],color=rgba_red[service_agent_idx])
						
						# curr_ax.scatter(locs[~service_agent_idx,0],locs[~service_agent_idx,1],color=agent_colors[1][~service_agent_idx])
						# curr_ax.scatter(locs[~service_agent_idx,0],locs[~service_agent_idx,1],color=agent_colors[0][service_agent_idx])

						service_agent_idx = np.asarray(results["agents_operation"][timestep],dtype=bool)
						x = np.linspace(0,1,param["ni"])
						curr_ax.scatter(locs[service_agent_idx,0],locs[service_agent_idx,1],c=x[service_agent_idx],cmap=agent_colors[1])
						curr_ax.scatter(locs[~service_agent_idx,0],locs[~service_agent_idx,1],c=x[~service_agent_idx],cmap=agent_colors[0])

						# service_agent_idx = np.asarray(results["agents_operation"][timestep],dtype=bool)
						# curr_ax.scatter(locs[service_agent_idx,0],locs[service_agent_idx,1],c=x[service_agent_idx],cmap=plt.get_cmap('Oranges'))
						# curr_ax.scatter(locs[~service_agent_idx,0],locs[~service_agent_idx,1],c=x[~service_agent_idx],cmap=plt.get_cmap('Blues'))

				elif 'customer' in key:
					curr_ax.scatter(locs[:,0],locs[:,1],c=service_color)

		elif 'operation' in key:
			nmode = 2
			hist = np.zeros((nmode))
			for i_mode in range(nmode):
				hist[i_mode] = sum(results[key][timestep] == i_mode) / param["ni"]
			curr_ax.bar(range(nmode),hist)
			curr_ax.set_ylim([0,1])
			plt.xticks(range(nmode),param["mode_names"],rotation='vertical')
		
		if 'distribution' in key or 'location' in key or 'action' in key:

			if param["env_name"] in 'citymap':
				curr_ax.plot(city_boundary[:,0],city_boundary[:,1],linewidth=1)

			# create same axis
			curr_ax.set_xticks(param["env_x"]) 
			curr_ax.set_yticks(param["env_y"]) 
			# curr_ax.set_xticks([]) 
			# curr_ax.set_yticks([]) 
			curr_ax.set_xlim(param["env_xlim"])
			curr_ax.set_ylim(param["env_ylim"])
			curr_ax.set_aspect('equal')
			curr_ax.grid(True)

		plt.setp(curr_ax.get_xticklabels(), visible=False)
		plt.setp(curr_ax.get_yticklabels(), visible=False)
		axs.append(curr_ax)

	# colorbar
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(im, cax=cbar_ax)

	# title
	t = param["sim_times"][timestep]
	T = param["sim_times"][-1]
	fig.suptitle('{} at t/T = {}/{}'.format(controller_name,t,T))
	return axs

def animate_plot(controller_name,sim_result,timestep,fig):
	# input: 
	# 	- sim_result is a dictionary of results for some controller
	# 	- timestep is when to plot

	param = sim_result["param"]
	state_keys = param["state_keys"]
	plot_keys = param["plot_keys"]

	axs = []
	ncol = 2
	nrow = 2 #int(np.floor(len(state_keys)/ncol)) + 1
	plt.subplots_adjust(top=0.8,hspace=0.25)
	myfontsize=14

	for i_key, key in enumerate(plot_keys):
		num = i_key + 1
		curr_ax = fig.add_subplot(nrow,ncol,num)
		# curr_ax.set_title(key,fontsize=16)
		
		if key in ['customers_location']:

			locs = sim_result[key][timestep] 
			if np.size(locs) > 0:
				curr_ax.scatter(locs[:,0],locs[:,1])

			curr_ax.set_title('Customers Location',fontsize=myfontsize)

		elif key in ['agents_value_fnc_distribution']:

			agent_idx = 0
			im_to_plot = sim_result[key][timestep][agent_idx]
			im = curr_ax.imshow(sim_to_im_coordinate(im_to_plot),
				vmin=0,vmax=1,cmap='gray_r',
				extent=[param["env_xlim"][0],param["env_xlim"][1],param["env_ylim"][0],param["env_ylim"][1]])

			curr_ax.set_title('Value Function',fontsize=myfontsize)
			
		elif key in ['agents_location']:
			
			locs = sim_result[key][timestep] 
			free_idx = sim_result["agents_operation"][timestep]
			free_idx = np.asarray(free_idx,dtype=bool)
			
			if np.size(locs) > 0:
				# curr_ax.scatter(locs[~free_idx,0],locs[~free_idx,1],cmap=plt.get_cmap('Blues'))
				# curr_ax.scatter(locs[free_idx,0],locs[free_idx,1],cmap=plt.get_cmap('Oranges'))
				curr_ax.scatter(locs[~free_idx,0],locs[~free_idx,1])
				curr_ax.scatter(locs[free_idx,0],locs[free_idx,1])				

			curr_ax.set_title('Taxi Locations',fontsize=myfontsize)

		elif key in ['reward']:
			
			times = sim_result["times"][0:timestep]
			rewards = np.cumsum(sim_result["rewards"][0:timestep])
			curr_ax.plot(times,rewards)
			curr_ax.set_xlabel('time')
			curr_ax.set_ylabel('reward')
			curr_ax.set_xlim([0,sim_result["times"][-1]])
			curr_ax.set_ylim([sum(sim_result["rewards"]),1])

		if 'distribution' in key or 'location' in key or 'action' in key:
			curr_ax.set_xticks(param["env_x"]) 
			curr_ax.set_yticks(param["env_y"]) 
			curr_ax.set_xlim(param["env_xlim"])
			curr_ax.set_ylim(param["env_ylim"])
			curr_ax.set_aspect('equal')
			curr_ax.grid(True)
			
		plt.setp(curr_ax.get_xticklabels(), visible=False)
		plt.setp(curr_ax.get_yticklabels(), visible=False)
		axs.append(curr_ax)

	# title
	t = param["sim_times"][timestep]
	T = param["sim_times"][-1]
	fig.suptitle('{} at t/T = {}/{} \n'.format(controller_name,t,T))
	return axs	

def plot_distribution_over_time(results,env):
	for timestep,time in enumerate(results.times):
		plot_distribution(results, timestep, env)

def sim_plot_over_time(controller_name,sim_result,times):
	
	param = sim_result["param"]
	agent_colors = get_agent_colors(param["ni"])
	if param["env_name"] in 'citymap':
		city_boundary = make_city_boundary(param)
		for timestep,time in times:
			fig = plt.figure()
			sim_plot(controller_name, sim_result, timestep, fig=fig, city_boundary=city_boundary)
	else:
		for timestep,time in times:
			fig = plt.figure()
			sim_plot(controller_name, sim_result, timestep, fig=fig, agent_colors=agent_colors)


def render(controller_name,sim_result,timestep):


	fig = plt.figure()
	ax1 = fig.add_subplot(1,2,1)
	ax2 = fig.add_subplot(1,2,2)

	# 
	param = sim_result["param"]	
	agent_colors = get_agent_colors(param["ni"])

	# subplot 1: agents/customers

	# plot customers
	customer_color = 'green'
	customer_locations = sim_result["customers_location"][timestep]
	print('customer_locations:',customer_locations)
	if len(customer_locations) > 0:
		ax1.scatter(customer_locations[:,0],customer_locations[:,1],color=customer_color)

	# plot servicing agents and free agents
	service_agent_idx = np.asarray(sim_result["agents_operation"][timestep],dtype=bool)
	x = np.linspace(0,1,param["ni"])
	agent_locs = sim_result["agents_location"][timestep]
	ax1.scatter(agent_locs[service_agent_idx,0],agent_locs[service_agent_idx,1],c=x[service_agent_idx],cmap=agent_colors[1])
	ax1.scatter(agent_locs[~service_agent_idx,0],agent_locs[~service_agent_idx,1],c=x[~service_agent_idx],cmap=agent_colors[0])

	# subplot 2: value function 
	agent_idx = 0 
	im_to_plot = sim_result["agents_value_fnc_distribution"][timestep][agent_idx]
	im = ax2.imshow(sim_to_im_coordinate(im_to_plot),
		vmin=0,vmax=1,cmap='gray_r',
		extent=[param["env_xlim"][0],param["env_xlim"][1],param["env_ylim"][0],param["env_ylim"][1]])

	# some axis stuff 
	ax1.set_xticks(param["env_x"]) 
	ax1.set_yticks(param["env_y"]) 
	ax1.set_xlim(param["env_xlim"])
	ax1.set_ylim(param["env_ylim"])
	ax1.set_aspect('equal')
	ax1.grid(True)
	plt.setp(ax1.get_xticklabels(), visible=False)
	plt.setp(ax1.get_yticklabels(), visible=False)

	ax2.set_xticks(param["env_x"]) 
	ax2.set_yticks(param["env_y"]) 
	ax2.set_xlim(param["env_xlim"])
	ax2.set_ylim(param["env_ylim"])
	ax2.set_aspect('equal')
	ax2.grid(True)	
	plt.setp(ax2.get_xticklabels(), visible=False)
	plt.setp(ax2.get_yticklabels(), visible=False)





def get_agent_colors(ni):

	import matplotlib.pyplot as plt
	import matplotlib.colors as colors
	import numpy as np

	def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
		new_cmap = colors.LinearSegmentedColormap.from_list(
			'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
			cmap(np.linspace(minval, maxval, n)))
		return new_cmap

	min_v_blue = 0.2
	max_v_blue = 0.6
	min_v_orange = 0.5
	max_v_orange = 0.6

	blue_cmap = plt.get_cmap('Blues')
	new_blue_cmap = truncate_colormap(blue_cmap, min_v_blue, max_v_blue)
	orange_cmap = plt.get_cmap('Oranges')
	new_orange_cmap = truncate_colormap(orange_cmap, min_v_orange, max_v_orange)
	

	# cvals  = [0,1] #[-2., -1, 2]
	# blue_colors = ['#666FF','#666CC'] # ["red","violet","blue"]
	# orange_colors = ['#FF9966','#FF6633']

	# norm=plt.Normalize(min(cvals),max(cvals))
	# blue_tuples=list(zip(map(norm,cvals), blue_colors))
	# hex doesnt work here 
	# blue_cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", blue_tuples)

	# norm=plt.Normalize(min(cvals),max(cvals))
	# orange_tuples=list(zip(map(norm,cvals), orange_colors))
	# orange_cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", orange_tuples)

	return [new_blue_cmap,new_orange_cmap]

	# blue_colors = []
	# orange_colors = []
	# # indices to step through colormap 
	# x = np.linspace(0.0, 1.0, ni)
	# agent_colors.append()


def sim_to_im_coordinate(im):
	# im coordinates
	im = im.T
	im = np.flipud(im)
	return im 

def ave_actions_vector_plot(param,im_ave_vec_action,xlim=None,ylim=None,fig=None,ax=None):
	# plots average action at each state as a vector
	# input:
	# 	- im_vec_action (nx,ny,2)
	# 	- plot param

	dx = param["env_dx"]
	env_x = np.asarray(param["env_x"])
	env_y = np.asarray(param["env_y"])

	X,Y = np.meshgrid(env_x + dx/2,env_y + dx/2)
	U = np.zeros(X.shape)
	V = np.zeros(X.shape)
	C = np.zeros(X.shape)

	for i_x,_ in enumerate(env_x):
		for i_y,_ in enumerate(env_y):
			U[i_y,i_x] = im_ave_vec_action[i_x,i_y,0]
			V[i_y,i_x] = im_ave_vec_action[i_x,i_y,1]
			C[i_y,i_x] = np.linalg.norm(im_ave_vec_action[i_x,i_y,:])

	# normalize arrow length
	idx = np.nonzero(U**2 + V**2)
	U[idx] = U[idx] / np.sqrt(U[idx]**2 + V[idx]**2);
	V[idx] = V[idx] / np.sqrt(U[idx]**2 + V[idx]**2);

	im = ax.quiver(X[idx],Y[idx],U[idx],V[idx],C[idx]) #,scale_units='xy')

def many_sim_plot(many_sim_results,many_sim_dict,sim_key):

	# plots total reward for different parameters 
	# many_sim_results is a list of sim_results, for each many_sim parameter
	# 	sim_results is a dictionary of [key,value] = [controller_name,sim_result]
	# 		sim_result is a dictionary of [key,value] output by sim.run

	# use some param 
	sim_results = many_sim_results[0]
	controller_names = sim_results.keys()
	for controller_name in controller_names: 
		param = sim_results[controller_name]["param"]
		break

	# assign markers by controller
	marker_dict = dict()
	color_dict = dict()
	marker_count = 0
	for controller_name in controller_names:
		marker_dict[controller_name] = param["plot_markers"][marker_count]
		color_dict[controller_name] = param["plot_colors"][marker_count]
		marker_count += 1 

	# unpack many sim dict  parameters 
	for varied_parameter, varied_parameter_values in many_sim_dict.items():

		dict_of_lsts = dict()
		for controller_name in controller_names:
			dict_of_lsts[controller_name] = []

		for (sim_results, varied_parameter_value) in zip(many_sim_results, varied_parameter_values):
			for controller_name, sim_result in sim_results.items():
				dict_of_lsts[controller_name].append(sim_result[sim_key])

		fig1,ax1 = make_fig()
		for controller_name, lst in dict_of_lsts.items():
			ax1.plot(varied_parameter_values, lst, marker=marker_dict[controller_name],
				color=color_dict[controller_name],label=controller_name)
		ax1.set_xlabel(varied_parameter)
		ax1.set_ylabel(sim_key)
		ax1.legend()

def plot_cumulative_reward(sim_results):
	
	# make dict of lsts of sim_results by controller
	sim_results_by_controller = dict()
	for sim_result in sim_results:
		if not sim_result["controller_name"] in sim_results_by_controller.keys():
			sim_results_by_controller[sim_result["controller_name"]] = [] 

	for sim_result in sim_results:
		sim_results_by_controller[sim_result["controller_name"]].append(sim_result)

	fig,ax = make_fig()
	for controller_name,sim_results in sim_results_by_controller.items():
		plot_cumulative_reward_w_trials(sim_results,ax=ax,label=controller_name)

	ax.legend()
	# ax.set_ylabel('Cumulative Reward')
	ax.set_ylabel('Cumulative Customer Waiting Time')
	ax.set_xlabel('Time')


def plot_cumulative_reward_w_trials(sim_results,ax=None,label=None):
	# input: 
	# 	- sim results is a list of sim_result dicts with same controller

	# extract rewards
	rewards = []
	for sim_result in sim_results:
		rewards.append(sim_result["rewards"]) 

	if False:
		# old 
		rewards = np.asarray(rewards)
	else:
		# new
		rewards = -1*np.asarray(rewards)
	
	rewards = np.cumsum(rewards,axis=1)

	# get some general parameters
	times = sim_result["times"]
	marker_dict,color_dict = get_marker_color_dicts(sim_result["param"])

	# mean and std
	mean = np.mean(rewards,axis=0)
	std = np.std(rewards,axis=0)

	# plot
	ax.plot(times,mean,label=label,color=color_dict[label]) 
	ax.fill_between(times,mean-std,mean+std,facecolor=color_dict[label],linewidth=1e-3,alpha=0.2)
	ax.grid(True)

def get_marker_color_dicts(param):
	# assign markers by controller
	marker_dict = dict()
	color_dict = dict()
	count = 0

	for key in param["controller_names"]:
		marker_dict[key] = param["plot_markers"][count]
		color_dict[key] = param["plot_colors"][count]
		count += 1 
	return marker_dict,color_dict



def plot_runtime_vs_number_of_agents(sim_results):

	ni_lst = []
	controller_name_lst = []
	for sim_result in sim_results:
		if not sim_result["controller_name"] in controller_name_lst:
			controller_name_lst.append(sim_result["controller_name"])
		if not sim_result["param"]["ni"] in ni_lst:
			ni_lst.append(sim_result["param"]["ni"])

	marker_dict,color_dict = get_marker_color_dicts(sim_result["param"])

	runtimes_by_controller_and_ni_dict = dict() # dict of dict of lsts
	for controller_name in controller_name_lst:
		runtimes_by_controller_and_ni_dict[controller_name] = dict()
		for ni in ni_lst:
			runtimes_by_controller_and_ni_dict[controller_name][ni] = []

	for sim_result in sim_results:
		runtimes_by_controller_and_ni_dict[sim_result["controller_name"]][sim_result["param"]["ni"]].append(sim_result["sim_run_time"])

	fig,ax = make_fig()
	for controller_name,controller_dict in runtimes_by_controller_and_ni_dict.items():
		plot_ni = []
		plot_mean = []
		plot_std = []
		for ni,rt_values in controller_dict.items():
			plot_ni.append(ni)
			plot_mean.append(np.mean(rt_values))
			plot_std.append(np.std(rt_values))
		
		# as numpy
		plot_ni = np.asarray(plot_ni)
		plot_mean = np.asarray(plot_mean)
		plot_std = np.asarray(plot_std)

		# sorted
		idxs = plot_ni.argsort()
		plot_ni = plot_ni[idxs]
		plot_mean = plot_mean[idxs]
		plot_std = plot_std[idxs]

		ax.plot( plot_ni, plot_mean, 
			color = color_dict[controller_name], 
			marker = marker_dict[controller_name], 
			label = controller_name)
		
		ax.errorbar(plot_ni, plot_mean, yerr=plot_std, color = color_dict[controller_name], linewidth=1e-3)
		ax.fill_between(plot_ni,plot_mean-plot_std,plot_mean+plot_std,facecolor=color_dict[controller_name],linewidth=1e-3,alpha=0.2)
	ax.legend()

def make_city_boundary(param):
	
	import shapefile
	from shapely.geometry import shape 
	from shapely.ops import unary_union, nearest_points, split
	from shapely.geometry import Point, Polygon, LineString

	# read shapefile 
	shp = shapefile.Reader(param["shp_path"])
	print(param["shp_path"])
	shapes = [shape(polygon) for polygon in shp.shapes()]

	# union shape
	print('   merge polygons in mapfile...')
	union = unary_union(shapes)
	if union.geom_type == 'MultiPolygon':
		city_polygon = union[0]			
		for polygon in union:
			if polygon.area > city_polygon.area:
				city_polygon = polygon
	else:
		city_polygon = union

	# make boundary
	x,y = city_polygon.exterior.coords.xy
	city_boundary = np.asarray([x,y]).T # [npoints x 2]
	return city_boundary

def plot_q_error(sim_results):

	# plot e(t), where e = (\|q^i - q^b\|)/n_i (average taxi error)
	# 	- for different controllers
	# 	- for different trials 

	# use dict of lsts of np arrays

	# init dict of lst
	q_values_by_controller_dict = dict() 
	for sim_result in sim_results:
		if not sim_result["controller_name"] in q_values_by_controller_dict.keys():
			q_values_by_controller_dict[sim_result["controller_name"]] = []

	marker_dict,color_dict = get_marker_color_dicts(sim_result["param"])
	times = sim_result["times"]

	# print('nt:',sim_result["param"]["sim_nt"])
	print('ni:',sim_result["param"]["ni"])
	print('nq:',sim_result["param"]["nq"])
	print('n_trials:',sim_result["param"]["n_trials"])

	# load np arrays of shape: [(nt-1), ni, nq]
	for sim_result in sim_results:
		# print('sim_result["agents_q_value"].shape',sim_result["agents_q_value"].shape)
		# correct! 
		q_values_by_controller_dict[sim_result["controller_name"]].append(sim_result["agents_q_value"])

	# get bellman soln, should be in 
	for controller_name, controller_values in q_values_by_controller_dict.items():
		if 'bellman' in controller_name:
			q_bellman = controller_values
			break 
	
	# q_bellman in [ntrials, nt, ni, nq]
	q_bellman = np.asarray(q_bellman)

	fig,ax = make_fig()
	for controller_name,controller_values in q_values_by_controller_dict.items():

		if not 'bellman' in controller_name and not 'rhc' in controller_name:

			# controller_values is a lst of np arrays of [nt, ni, nq]
			# controller_values_np is an np arrays of [ntrials, nt, ni, nq]
			controller_values_np = np.asarray(controller_values)
			ntrials = controller_values_np.shape[0]

			# take average q error per (s,a) pair 
			# [ntrials,nt,ni,nq] -> [ntrials,nt,ni]
			nq = sim_result["param"]["nq"]
			error = np.linalg.norm((controller_values_np - q_bellman)/q_bellman/nq, axis=3)
			# error = np.linalg.norm((controller_values_np - q_bellman), axis=3)

			# average across agents 
			# [ntrials,nt,ni] -> [ntrials,nt]
			error = np.mean(error,axis=2)
			
			plot_trial_statistics_on = False
			if plot_trial_statistics_on:
				# take statistics across trials 
				# [ntrials,nt] -> [nt]
				error_mean = np.mean(error,axis=0)
				error_std = np.std(error,axis=0)

				ax.plot(times,error_mean,
					color=color_dict[controller_name], 
					label=controller_name)
				ax.errorbar(times, error_mean, 
					yerr=error_std, 
					color=color_dict[controller_name], 
					linewidth=1e-3)
				ax.fill_between(times,error_mean-error_std,error_mean+error_std,
					facecolor=color_dict[controller_name],
					linewidth=1e-3,alpha=0.2)
			else:
				trial_num = 1
				error = error[trial_num,:]
				ax.plot(times,error,
					color=color_dict[controller_name], 
					label=controller_name)

	ax.set_title('Q MSE')
	ax.legend()

def plot_runtime_vs_state_space(sim_results):

	ncell_lst = []
	controller_name_lst = []
	for sim_result in sim_results:
		if not sim_result["controller_name"] in controller_name_lst:
			controller_name_lst.append(sim_result["controller_name"])
		if not sim_result["param"]["env_ncell"] in ncell_lst:
			ncell_lst.append(sim_result["param"]["env_ncell"])

	marker_dict,color_dict = get_marker_color_dicts(sim_result["param"])

	runtimes_by_controller_and_ni_dict = dict() # dict of dict of lsts
	for controller_name in controller_name_lst:
		runtimes_by_controller_and_ni_dict[controller_name] = dict()
		for ncell in ncell_lst:
			runtimes_by_controller_and_ni_dict[controller_name][ncell] = []

	for sim_result in sim_results:
		runtimes_by_controller_and_ni_dict[sim_result["controller_name"]][sim_result["param"]["env_ncell"]].append(sim_result["sim_run_time"])

	fig,ax = make_fig()
	for controller_name,controller_dict in runtimes_by_controller_and_ni_dict.items():
		plot_ncell = []
		plot_mean = []
		plot_std = []
		for ncell,rt_values in controller_dict.items():
			plot_ncell.append(ncell)
			plot_mean.append(np.mean(rt_values))
			plot_std.append(np.std(rt_values))
		
		# as numpy
		plot_ncell = np.asarray(plot_ncell)
		plot_mean = np.asarray(plot_mean)
		plot_std = np.asarray(plot_std)

		# sorted
		idxs = plot_ncell.argsort()
		plot_ncell = plot_ncell[idxs]
		plot_mean = plot_mean[idxs]
		plot_std = plot_std[idxs]

		ax.plot( plot_ncell, plot_mean, 
			color = color_dict[controller_name], 
			marker = marker_dict[controller_name], 
			label = controller_name)
		
		ax.errorbar(plot_ncell, plot_mean, yerr=plot_std, color = color_dict[controller_name], linewidth=1e-3)
		ax.fill_between(plot_ncell,plot_mean-plot_std,plot_mean+plot_std,facecolor=color_dict[controller_name],linewidth=1e-3,alpha=0.2)
	ax.legend()