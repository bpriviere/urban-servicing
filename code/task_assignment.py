

import numpy as np 
import cvxpy as cp 

import utilities

def binary_log_learning(env,agents):
	
	n_agents = len(agents)
	cell_assignments = []

	# assign random actions to each agent
	for agent in agents:
		random_action = np.random.randint(0,env.param.env_naction)
		agent.action = random_action

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
		
		# propose a random action 
		action_p = np.random.randint(0,env.param.env_naction)
		while action_p == agent.action:
			action_p = np.random.randint(0,env.param.env_naction)

		# calculate marginal utility of local action sets
		# NOTE: we calculate marginal cost instead, and flip sign convention
		J = calc_J(env,agent,agent.action,agents)
		J_p = calc_J(env,agent,action_p,agents)

		# assign action probability with binary log-linear learning algorithm
		# P_i = np.exp(J/tau)/(np.exp(J/tau) + np.exp(J_p/tau)) check more numerically stable ver below 
		P_inv = np.exp((J_p/tau)-(J/tau))+1
		P_i = 1/P_inv

		# print('agent.i: ', agent.i)		
		# print('agent.action: ', agent.action)
		# print('action_p: ', action_p)
		# print('J: ',J)
		# print('J_p: ', J_p)
		# print('P_i: ', P_i)

		# check convergence
		rand = np.random.random()
		if rand > P_i:
			same_action_count[a_idx] += 1
			if same_action_count[a_idx] > env.param.ta_converged:
				converged[a_idx] = True
		else:
			agent.action = action_p 
			same_action_count[a_idx] = 0

		count += 1 
		if count >= k_count*env.param.ta_tau_decay_threshold:
			tau = tau*env.param.ta_tau_decay
			k_count += 1 

		if count >= 10000:
			print('blll not converging')
			break
			exit('blll not converging')

	print('blll count: ',count)
	for agent in agents:
		cell_assignments.append((agent,agent.action))
	return cell_assignments


def calc_J(env,agent_i,action_i,agents):

	# local state indices
	local = []
	P = utilities.get_MDP_P(env)
	s = utilities.coordinate_to_cell_index(agent_i.x,agent_i.y)
	for a in range(env.param.env_naction):
		next_state = np.where(P[a,s,:] == 1)[0][0]
		if not next_state in local:
			local.append(next_state)

	H = make_H(env,agents)
	A = np.zeros((env.param.env_naction*len(agents)))
	for j,agent_j in enumerate(agents):
		if agent_i is agent_j:
			idx = j*env.param.env_naction + action_i
		else:
			idx = j*env.param.env_naction + agent_j.action 
		A[idx] = 1

	local_HA = np.matmul(H,A)[local]
	local_S = make_S(env,agents)[local]
	local_V = agent_i.v[local]
	J = np.linalg.norm(local_S + local_HA - local_V)
	return J 


def centralized_linear_program(env,agents):
	# input: 
	#    - env
	#    - agents: list of idle agents
	# ouput: 
	#    - cell_assignments: integer variables corresponding to tabular actions 
	
	cell_assignments = []
	if not len(agents) == 0:
		H = make_H(env,agents)
		S = make_S(env,agents)
		V = env.agents[0].v
		a = cp.Variable(env.param.env_naction*len(agents), integer=True)

		obj = cp.Minimize( cp.sum_squares( S + H@a - V ))
		# constrain action value to be zero or one 
		constr = [a <= 1, a >= 0]
		# every agent only takes one action
		for i in range(len(agents)):
			idx = np.arange(0,env.param.env_naction) + i*env.param.env_naction
			constr.append(sum(a[idx]) == 1)
		prob = cp.Problem(obj, constr)
		prob.solve(verbose = True, solver = cp.GUROBI)

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
		state = utilities.coordinate_to_cell_index(agent.x,agent.y)
		S[state] += 1 
	return S/len(agents)


def make_H(env,agents):
	
	n_agents = len(agents)
	H = np.zeros((env.param.env_ncell,env.param.env_naction*n_agents))
	P = utilities.get_MDP_P(env)
	for step,agent in enumerate(agents):
		idx = step*env.param.env_naction
		state = utilities.coordinate_to_cell_index(agent.x,agent.y)
		# print('agent.x: ', agent.x)
		# print('agent.y: ', agent.y)
		for a in range(env.param.env_naction):
			# print('P: ',P)
			# print('a: ',a)
			# print('state: ', state)
			next_state = np.where(P[a,state,:] == 1)[0][0]
			H[state,idx+a] = -1
			H[next_state,idx+a] = 1	
	# normalize
	H = H/env.param.ni
	return H 


