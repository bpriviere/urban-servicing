

# my packages 
from param import Param
from gridworld import GridWorld
from citymap import CityMap
import datahandler
import plotter 

import matplotlib.pyplot as plt

def main():

	param = Param()
	env = CityMap(param)

	# init datasets
	if param.make_dataset_on:
		print('   making dataset...')
		train_dataset, test_dataset = datahandler.make_dataset(env)
		datahandler.write_dataset(env, train_dataset, test_dataset)
	print('   loading dataset...')
	datahandler.load_dataset(env)

	env.init_map()

	# fig,ax=plotter.make_fig()
	# env.print_map(fig=fig,ax=ax)

	train_heatmap = env.heatmap(env.train_dataset)
	test_heatmap = env.heatmap(env.test_dataset)

	fig = plt.figure()
	gs = fig.add_gridspec(2,2)
	ax1 = fig.add_subplot(gs[0, 0])
	ax2 = fig.add_subplot(gs[0, 1])
	ax3 = fig.add_subplot(gs[1, :])

	plotter.plot_heatmap(env,train_heatmap,ax1)
	plotter.plot_heatmap(env,test_heatmap,ax2)
	plotter.plot_cell_demand_over_time(env, ax3)
	
	# ax1.set_ylim(bottom=41.85)
	# ax2.set_ylim(bottom=41.85)
	# ax1.set_xlim(left=-87.75) 
	# ax2.set_xlim(left=-87.75)

	ax1.axis("off") 
	ax2.axis("off")
	ax1.set_title('October 29, 2016')
	ax2.set_title('October 30, 2016')

	# ax1.get_yaxis().set_visible(False)
	# ax2.get_xaxis().set_visible(False)
	# ax2.get_yaxis().set_visible(False)


	print('saving and opening figs...')
	plotter.save_figs(param.plot_fn)
	plotter.open_figs(param.plot_fn)
	

if __name__ == '__main__':
	main()