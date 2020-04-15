
# standard package
import glob
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.animation as manimation

# my packages
from param import Param 
from plotter import sim_plot, animate_plot_2
import datahandler

plt.rcParams.update({'font.size': 8})
plt.rcParams['lines.linewidth'] = 4

class Animation:
	def __init__(self,sim_result,param):

		self.fig = plt.figure() 
		
		temp_on = False
		if temp_on:
			nframes = 5
		else:
			nframes = len(sim_result["times"])


		self.axs = []
		self.anim = animation.FuncAnimation(self.fig, self.animate_func,fargs=[sim_result],
			frames=nframes,init_func=self.init_func)

	def show(self):
		plt.show()

	def animate_func(self, i, sim_result):
		controller_name = sim_result["controller_name"]
		timestep = i 
		self.init_func()
		self.axs = animate_plot_2(controller_name,sim_result,timestep,fig=self.fig)
		return self.axs

	def init_func(self):
		self.fig.clf()
		for ax in self.axs:
			ax.cla()
		return []

	def save(self,fn,fps,resolution):
		self.anim.save(
			fn,
			"ffmpeg",
			fps=fps,
			dpi=resolution)


if __name__ == "__main__":

	# some param
	param = Param()
	fn = 'video.mp4'
	fps = 5
	resolution = 100
	save = True

	result_dirs = ['../current_results/*']

	# load sim sim_results 
	sim_results = [] # lst of dicts
	print('loading sim results...')
	for result_dir in result_dirs:
		print('result_dir: ',result_dir)
		for sim_result_dir in glob.glob(result_dir):
			print('sim_result_dir: ',sim_result_dir)
			if not 'plots.pdf' in sim_result_dir:
				sim_result = datahandler.load_sim_result(sim_result_dir)
				sim_results.append(sim_result)

	print('making animation...')
	animation = Animation(sim_results[0],param)
	
	if save:
		print('saving animation...')
		animation.save(fn,fps,resolution)
	else:
		print('showing animation...')
		animation.show()
