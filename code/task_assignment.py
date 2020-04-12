

import numpy as np 
import cvxpy as cp 

def binary_log_learning(env,agents):
	
	n_agents = len(agents)
	cell_assignments = []

	# assign best/random actions to each agent
	for agent in agents:
		random_action = np.random.randint(0,env.param.env_naction)

		state = env.coordinate_to_cell_index(agent.x,agent.y)
		local = env.get_local_states(state)
		best_action = np.argmax(agent.v[local])

		if False:
			agent.cell_action = random_action
		else:
			agent.cell_action = best_action

	converged = np.zeros(n_agents,dtype=bool)
	same_action_count = np.zeros(n_agents)
	count = 0 
	k_count = 1
	tau = env.param.ta_tau
	
	H = make_H(env,agents)
	A = make_A(env,agents)
	S = make_S(env,agents)
	print('   blll...')
	while not all(converged):
		
		# pick a random non-converged agent 
		c_idx = np.random.randint(0,sum(np.logical_not(converged)))
		a_idx = np.where(converged == 0)[0][c_idx]
		agent = agents[a_idx]
		state = env.coordinate_to_cell_index(agent.x,agent.y)
		next_state = env.get_next_state(state,agent.cell_action)
		
		# propose a random action 
		action_p = np.random.randint(0,env.param.env_naction)
		next_state_p = env.get_next_state(state,action_p)
		while next_state == next_state_p:
			action_p = np.random.randint(0,env.param.env_naction)
			next_state_p = env.get_next_state(state,action_p)

		H_p = make_H_p(H,agent,action_p,env,a_idx,n_agents)
		A_p = make_A_p(A,agent,action_p,env,a_idx)
		
		# calculate marginal utility of local action sets
		J = calc_J(env,agent,H,A,S)
		J_p = calc_J(env,agent,H_p,A_p,S)

		# assign action probability with binary log-linear learning algorithm
		# P_i = np.exp(J/tau)/(np.exp(J/tau) + np.exp(J_p/tau)) 
		P_inv = np.exp((J_p/tau)-(J/tau))+1
		P_i = 1/P_inv

		# check convergence
		rand = np.random.random()
		# NOTE: we calculate marginal cost instead, and flip sign convention
		if rand > P_i:
			same_action_count[a_idx] += 1
			if same_action_count[a_idx] > env.param.ta_converged:
				converged[a_idx] = True

		else:
			agent.cell_action = action_p 
			same_action_count[a_idx] = 0

			H = H_p
			A = A_p

		count += 1 

		# to help convergence
		if count >= k_count*env.param.ta_tau_decay_threshold:
			tau = tau*env.param.ta_tau_decay
			k_count += 1 

		if count >= np.max((500,env.param.ta_iter_lim_per_agent*env.param.ni)):
			print('   blll not converging: {}'.format(count))
			break

	if True:
		print('   blll count: ',count)
		print('   n free agent: ', len(agents))

	for agent in agents:
		cell_assignments.append((agent,agent.cell_action))
	return cell_assignments

def calc_J(env,agent_i,H,A,S):
	# input: 
	# 	H, swarm distribution transition matrix : numpy in [ncell x naction*n_agents]
	# 	A, joint action matrix : numpy in [env_naction*ni x 1]


	# get state and local state 	
	state = env.coordinate_to_cell_index(agent_i.x,agent_i.y)
	local_states, local_actions = env.get_local_transitions(state)

	state_action_pairs = []
	for action in local_actions:
		state_action_pairs.append((state,action))

	# get policy 
	pi_local = env.local_boltzmann_policy(agent_i.q, state_action_pairs)

	# get local agent distribution
	S_tp1_local = S[local_states] + np.matmul(H[local_states,:],A)
	S_tp1_local /= sum(S_tp1_local)

	# marginal cost 
	J = np.linalg.norm(S_tp1_local - pi_local)

	return J 


# def centralized_linear_program(env,agents):
# 	# input: 
# 	#    - env
# 	#    - agents: list of free agents
# 	# ouput: 
# 	#    - cell_assignments: integer variables corresponding to tabular actions that move agent to cell 
	
# 	cell_assignments = []
# 	if not len(agents) == 0:
# 		H = make_H(env,agents)
# 		S = make_S(env,agents)

# 		# V = env.value_to_probability(env.agents[0].v)
# 		V = env.value_to_probability(env.q_value_to_value_fnc(env.agents[0].q))
# 		a = cp.Variable(env.param.env_naction*len(agents), integer=True)
# 		obj = cp.Minimize( cp.sum_squares( S + H@a - V ))

# 		# constrain action value to be between zero or one 
# 		constr = [a <= 1, a >= 0]

# 		# every agent only takes one action forces a to be zero or one
# 		for i in range(len(agents)):
# 			idx = np.arange(0,env.param.env_naction) + i*env.param.env_naction
# 			constr.append(sum(a[idx]) == 1)
# 		prob = cp.Problem(obj, constr)

# 		# prob.solve(verbose = True, solver = cp.GUROBI)
# 		prob.solve(verbose = False, solver = cp.GUROBI)

# 		a = np.array(a.value,dtype=bool)
# 		a = a.reshape((len(agents),env.param.env_naction))
		
# 		# print(a)
# 		# exit()
# 		for i in range(len(agents)):
# 			agent = agents[i]
# 			action = np.where(a[i,:]==True)[0][0]
# 			cell_assignments.append((agent,action))

# 	return cell_assignments


def make_S(env,agents):
	# current swarm distribution, unnormalized 

	S = np.zeros((env.param.env_ncell))
	for agent in agents:
		state = env.coordinate_to_cell_index(agent.x,agent.y)
		S[state] += 1 
	return S

def make_A(env,agents):
	# joint action distribution  

	A = np.zeros((env.param.env_naction*len(agents)))
	for step,agent in enumerate(agents):
		idx = step*env.param.env_naction + agent.cell_action
		A[idx] = 1
	return A 

def make_H(env,agents):
	# swarm distribution transition matrix 
	
	n_agents = len(agents)
	H = np.zeros((env.param.env_ncell,env.param.env_naction*n_agents))
	
	for step,agent in enumerate(agents):
		idx = step*env.param.env_naction
		state = env.coordinate_to_cell_index(agent.x,agent.y)
		for action in range(env.param.env_naction):
			next_state = env.get_next_state(state,action)
			H[state,idx+action] = -1
			H[next_state,idx+action] = 1

	return H 

def make_H_p(H,agent,action,env,a_idx,n_agents):
	# alternative swarm distribution transition matrix  

	Hp = np.copy(H)

	# change cell 	
	idx = a_idx*env.param.env_naction
	state = env.coordinate_to_cell_index(agent.x,agent.y)
	next_state = env.get_next_state(state,action)
	Hp[state,idx+action] = -1
	Hp[next_state,idx+action] = 1

	return Hp

def make_A_p(A,agent,action,env,a_idx):
	# alternative joint action distribution 

	Ap = np.copy(A) 

	# remove prev action
	idx = a_idx*env.param.env_naction + agent.cell_action
	Ap[idx] = 0 
	# add new action 
	idx = a_idx*env.param.env_naction + action 
	Ap[idx] = 1
	
	return Ap 