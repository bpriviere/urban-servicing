
import sys, os, shutil
import glob
import argparse
import datetime 



def main(name):
	# take results from current_results directory and push into results
	# take data from current_data directory and push into data

	dt = datetime.date
	today = datetime.datetime.today()
	timestamp = '{0:04}-{1:02}-{2:02}_'.format(today.year, today.month, today.day)

	# save input data 
	src_data_dir = '../current_data'
	dest_data_dir = '../data/{}'.format(timestamp+name)
	shutil.copytree(src_data_dir, dest_data_dir)	

	# save results files 
	src_results_dir = '../current_results'
	dest_results_dir = '../results/{}'.format(timestamp+name)
	shutil.copytree(src_results_dir, dest_results_dir)

	# save plots 
	src_plot = 'plots.pdf'
	dest_plot = '../results/{}/plots.pdf'.format(timestamp+name)
	shutil.copy(src_plot, dest_plot)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', help='sim results name description')
	args = parser.parse_args()
	
	if args.n:
		main(args.n)
	else:
		exit('no argument given')
