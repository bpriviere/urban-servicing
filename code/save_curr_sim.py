
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

	temp_results_dir = '../current_results'
	new_results_dir = '../results/{}'.format(timestamp+name)
	shutil.copytree(temp_results_dir, new_results_dir)

	temp_data_dir = '../current_data'
	new_data_dir = '../data/{}'.format(timestamp+name)
	shutil.copytree(temp_data_dir, new_data_dir)	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', help='sim results name description')
	args = parser.parse_args()
	
	if args.n:
		main(args.n)
	else:
		exit('no argument given')
