

import numpy as np 
import cvxpy as cp 

def binary_log_learning(env,agents):
	
	n_agents = len(agents)
	cell_assignments = []

	# assign best/random actions to each agent
	for agent in agents:
		random_action = np.random.randint(0,env.param.env_naction)

		state = env.utilities.coordinate_to_cell_index(agent.x,agent.y)
		local = env.utilities.get_local_states(env,state)
		best_action = np.argmax(agent.v[local])

		# print()
		# print(local)
		# print(env.utilities.coordinate_to_cell_index(agent.x,agent.y))
		# print(best_action)
		# print(agent.v)
		# exit()

		if False:
			agent.cell_action = random_action
		else:
			agent.cell_action = best_action

	converged = np.zeros(n_agents,dtype=bool)
	same_action_count = np.zeros(n_agents)
	count = 0 
	k_count = 1
	tau = env.param.ta_tau
	while not all(converged):
		
		# pick a random non-converged agent 
		c_idx = np.random.randint(0,sum(np.logical_not(converged)))
		a_idx = np.where(converged == 0)[0][c_idx]
		agent = agents[a_idx]
		state = env.utilities.coordinate_to_cell_index(agent.x,agent.y)
		next_state = env.utilities.get_next_state(env,state,agent.cell_action)
		
		# propose a random action 
		action_p = np.random.randint(0,env.param.env_naction)
		next_state_proposed = env.utilities.get_next_state(env,state,action_p)
		while next_state == next_state_proposed:
			action_p = np.random.randint(0,env.param.env_naction)
			next_state_proposed = env.utilities.get_next_state(env,state,action_p)

			# print('env.utilities.coordinate_to_cell_index(agent.x,agent.y):',env.utilities.coordinate_to_cell_index(agent.x,agent.y))
			# print('agent.cell_action:',agent.cell_action)
			# print('action_p:',action_p)
			# print('next_state: ',next_state)
			# print('next_state_proposed: ',next_state_proposed)

		# calculate marginal utility of local action sets
		# NOTE: we calculate marginal cost instead, and flip sign convention
		J = calc_J(env,agent,agent.cell_action,agents)
		J_p = calc_J(env,agent,action_p,agents)

		# assign action probability with binary log-linear learning algorithm
		# P_i = np.exp(J/tau)/(np.exp(J/tau) + np.exp(J_p/tau)) 

		# is this more numerically stable ???
		P_inv = np.exp((J_p/tau)-(J/tau))+1
		P_i = 1/P_inv

		# print('agent.i: ', agent.i)		
		# print('agent.cell_action: ', agent.cell_action)
		# print('action_p: ', action_p)
		# print('J: ',J)
		# print('J_p: ', J_p)
		# print('P_i: ', P_i)
		# exit()

		# check convergence
		rand = np.random.random()
		if rand > P_i:
			same_action_count[a_idx] += 1
			if same_action_count[a_idx] > env.param.ta_converged:
				converged[a_idx] = True
		else:
			agent.cell_action = action_p 
			same_action_count[a_idx] = 0

		count += 1 
		if count >= k_count*env.param.ta_tau_decay_threshold:
			tau = tau*env.param.ta_tau_decay
			k_count += 1 

		if count >= np.max((500,env.param.blll_iter_lim_per_agent*env.param.ni)):
			print('blll not converging')
			break
			# print('J',J)
			# print('J_p',J_p)
			# print(converged)
			# exit('blll not converging')

	print('   blll count: ',count)
	print('   n free agent: ', len(agents))
	for agent in agents:
		cell_assignments.append((agent,agent.cell_action))
	return cell_assignments


def calc_J(env,agent_i,action_i,agents):

	# get local q values
	local_q = env.utilities.get_local_q_values(env,agent_i)
	local_q_dist = env.utilities.value_to_probability(local_q)

	# get local agent distribution
	state = env.utilities.coordinate_to_cell_index(agent_i.x,agent_i.y)
	local_s_idx = env.utilities.get_local_states(env,state)
	H = make_H(env,agents)
	A = get_joint_action(env,agent_i,action_i,agents)	
	local_agent_dist = np.matmul(H,A)[local_s_idx]

	# marginal cost 
	J = np.linalg.norm(local_agent_dist - local_q_dist)

	# debug 
	# local_v = env.utilities.value_to_probability(agent_i.v[local_s_idx])
	# J2 = np.linalg.norm(local_agent_dist - local_v)

	# print('J: ', J)
	# print('J2: ', J2)
	# exit()
	return J 


def get_joint_action(env,agent_i,action_i,agents):
	A = np.zeros((env.param.env_naction*len(agents)))
	for j,agent_j in enumerate(agents):
		if agent_i is agent_j:
			idx = j*env.param.env_naction + action_i
		else:
			idx = j*env.param.env_naction + agent_j.cell_action 
		A[idx] = 1
	return A 

def centralized_linear_program(env,agents):
	# input: 
	#    - env
	#    - agents: list of free agents
	# ouput: 
	#    - cell_assignments: integer variables corresponding to tabular actions that move agent to cell 
	
	cell_assignments = []
	if not len(agents) == 0:
		H = make_H(env,agents)
		S = make_S(env,agents)

		# V = env.utilities.value_to_probability(env.agents[0].v)
		V = env.utilities.value_to_probability(env.utilities.q_value_to_value_fnc(env,env.agents[0].q))

		# print('H',H)
		# print('S',S)
		# print('V',V)
		# exit()

		a = cp.Variable(env.param.env_naction*len(agents), integer=True)

		obj = cp.Minimize( cp.sum_squares( S + H@a - V ))

		# constrain action value to be between zero or one 
		constr = [a <= 1, a >= 0]

		# every agent only takes one action forces a to be zero or one
		for i in range(len(agents)):
			idx = np.arange(0,env.param.env_naction) + i*env.param.env_naction
			constr.append(sum(a[idx]) == 1)
		prob = cp.Problem(obj, constr)

		# prob.solve(verbose = True, solver = cp.GUROBI)
		prob.solve(verbose = False, solver = cp.GUROBI)

		a = np.array(a.value,dtype=bool)
		a = a.reshape((len(agents),env.param.env_naction))
		
		# print(a)
		# exit()
		for i in range(len(agents)):
			agent = agents[i]
			action = np.where(a[i,:]==True)[0][0]
			cell_assignments.append((agent,action))

	return cell_assignments


def make_S(env,agents):

	S = np.zeros((env.param.env_ncell))
	for agent in agents:
		state = env.utilities.coordinate_to_cell_index(agent.x,agent.y)
		S[state] += 1 
	return S/len(agents)


def make_H(env,agents):
	# output is a matrix in ncell, naction*n_agents
	# H[i,j] describes S[i] under A[j]
	
	n_agents = len(agents)
	H = np.zeros((env.param.env_ncell,env.param.env_naction*n_agents))
	
	# P = env.utilities.get_MDP_P(env)
	P = env.utilities.P
	
	for step,agent in enumerate(agents):
		idx = step*env.param.env_naction
		state = env.utilities.coordinate_to_cell_index(agent.x,agent.y)
		# print('agent.x: ', agent.x)
		# print('agent.y: ', agent.y)
		for action in range(env.param.env_naction):
			# print('P: ',P)
			# print('a: ',a)
			# print('state: ', state)
			next_state = env.utilities.get_next_state(env,state,action)
			H[state,idx+action] = -1
			H[next_state,idx+action] = 1
	# normalize
	# H = H/env.param.ni
	H /= n_agents
	return H 


