

import numpy as np 
import cvxpy as cp 

import utilities
from agent import IdleMove

def binary_log_learning(env):
	pass 


def centralized_linear_program(env,agents):
	# input: 
	#    - env
	#    - agents: list of idle agents
	# ouput: 
	#    - move_assignments: list of (agent,movement) tuples
	

	# first, solve LP to get a list of cell assignments
	n_idle_agents = len(agents)
	H = make_H(env,agents)
	print(H)
	exit()

	S = make_S(env)
	V = env.agents[0].V

	a = cp.Variable(n_actions_per_agent*n_idle_agents)

	obj = cp.Minimize( cp.sum_squares(a)
		)
	
	constraints = []
	for i in range(n_idle_agents):
		
		idx = agent.i + np.arange(0,n_actions_per_agent)
		constraints.append(
			a
			)

	prob = cp.Problem(obj, constraints)
	prob.solve(verbose = True) # , solver = cp.GUROBI)

	a = np.array(a)
	exit()

	# next, convert cell assignments to (x,y) movement assignments





def make_H(env,agents):
	
	n_agents = len(agents)
	H = np.zeros((env.param.env_ncell,env.param.env_naction*n_agents))
	P = utilities.get_MDP_P(env)
	for step,agent in enumerate(agents):
		idx = step*env.param.env_naction
		state = utilities.coordinate_to_cell_index(agent.x,agent.y)
		for a in range(env.param.env_naction):
			next_state = np.where(P[a,state,:] == 1)[0][0]
			H[state,idx+a] = -1
			H[next_state,idx+a] = 1	
	return H/env.param.ni 