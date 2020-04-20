

import numpy as np 
import random, math 
import cvxpy as cp 

def i_j_to_ij(env,i,j):
	return i*env.param.env_ncell + j 

def ij_to_i_j(env,ij):
	i = np.floor(ij / env.param.env_ncell)
	j = np.remainder(ij,i)
	return i,j

def centralized_linear_program(env,agents):
	# input: 
	#    - env
	#    - agents: list of free agents
	# ouput: 
	#    - cell_assignments: integer variables corresponding to tabular actions that move agent to cell 

	# decision variables 
	# 	x_t is number of taxis in each cell 
	# 	U_t[s,sp,t] is defined as cells in s going to cell sp at time t
	# 	U_t[ij,t] is flattened array bc cvxpy doesnt accept > 2d
	# ij = s*env_ncell + sp

	if not hasattr(env, 'w'):
		env.w = env.get_customer_demand(env.train_dataset,env.param.sim_times[0]) 

	cell_assignments = [] 

	if len(agents)>0:

		# 
		nfree = len(agents)
		
		# reward is in [nq x 1]
		r = agents[0].r
		q = agents[0].q

		# normalized prediction in [ns x 1]
		w = np.zeros((env.param.env_ncell,env.param.rhc_horizon))
		same_reward = True
		if same_reward: 
			for s in range(env.param.env_ncell):

				q_idx = env.sa_to_q_idx(s,0)
				w[s,:] = r[q_idx]

				# pi = env.global_boltzmann_policy(q)
				# w[s,:] = pi[q_idx]

			for t in range(env.param.rhc_horizon):
				w[:,t] = w[:,t] / sum(w[:,t])

		else:
			for t in range(env.param.rhc_horizon):
				w[:,t] = env.w

		# initial condition
		x0 = np.zeros((env.param.env_ncell))
		for agent in agents:
			s = env.coordinate_to_cell_index(agent.x,agent.y)
			x0[s] += 1

		# some preliminary manipulation 
		banned_idx = np.zeros((env.param.env_ncell,env.param.env_ncell**2))
		idxs_from_s = np.zeros((env.param.env_ncell,env.param.env_ncell**2))
		idxs_to_s = np.zeros((env.param.env_ncell,env.param.env_ncell**2))
		
		for s in range(env.param.env_ncell):
			for sp in range(env.param.env_ncell):
				idxs_from_s[s,i_j_to_ij(env,s,sp)] = 1 
				idxs_to_s[s,i_j_to_ij(env,sp,s)] = 1 
				banned_idx[s,i_j_to_ij(env,s,sp)] = 1

			local_states,local_actions = env.get_local_transitions(s)
			for sp in local_states:
				banned_idx[s, i_j_to_ij(env,s,sp)] = 0

		x_t = cp.Variable((env.param.env_ncell,env.param.rhc_horizon+1))
		U_t = cp.Variable((env.param.env_ncell**2,env.param.rhc_horizon))

		# init 
		cost = 0
		constr = []

		# initial condition 
		constr.append(x_t[:,0] == x0)

		# always greater than zero
		constr.append(U_t >= 0)

		# sum to ni 
		constr.append(cp.sum(U_t,axis=0)==nfree)

		for t in range(env.param.rhc_horizon):

			# conservation 1: taxis in 's' at next timestep = taxis curr in 's' + taxis going into cell 's'
			constr.append(idxs_to_s @ U_t[:,t] == x_t[:,t+1])

			# conservation 2: taxis leaving 's' less than or equal to taxis curr in 's' 
			constr.append(idxs_from_s @ U_t[:,t] == x_t[:,t])

			# dynamics: travel between banned indices are not allowed 
			constr.append(banned_idx @ U_t[:,t] == 0)

			# cost fnc:
			cost += env.param.mdp_gamma**t * cp.sum_squares(x_t[:,t+1]/nfree - w[:,t])

		# solve 
		obj = cp.Minimize(cost)
		prob = cp.Problem(obj, constr)
		# prob.solve(verbose=True, solver=cp.GUROBI)
		print('   solving LP...')
		prob.solve(verbose=False, solver=cp.GUROBI)
		# print('   solved LP')

		cell_assignments = U_to_cell_assignments(U_t.value[:,0],env,agents,idxs_from_s)
			
	return cell_assignments	


def U_to_cell_assignments(U,env,agents,idxs_from_s):

	# convert to integers
	cell_assignments = [] 

	random_round = False
	if random_round:
		cell_movement = np.zeros((U.shape))

		for ij in range(env.param.env_ncell**2):

			floor_u_ij = np.floor(U[ij])
			random = np.random.random()
			if random < U[ij] - floor_u_ij:
				cell_movement[ij] = floor_u_ij + 1 
			else:
				cell_movement[ij] = floor_u_ij 

		# 	print('floor_u_ij:',floor_u_ij)
		# 	print('U[ij]:',U[ij])
		# 	print('random:',random)
		# 	print('cell_movement[ij]:',cell_movement[ij])
		# exit()
	else:
		cell_movement = np.round(U)	
	
	cell_movement = cell_movement.astype(int)

	round_errors = 0
	for agent in agents:

		# get current cell, s 
		s = env.coordinate_to_cell_index(agent.x,agent.y)
		local_states,local_actions = env.get_local_transitions(s)

		# if there are still assignments leaving cell s 
		if sum(cell_movement[idxs_from_s[s,:] == 1]) > 0:

			# get first available cell 
			sp = np.nonzero(cell_movement[idxs_from_s[s,:] == 1])[0][0]

			# if transition state is valid 
			if sp in local_states:
				action = env.s_sp_to_a(s,sp)
				cell_movement[i_j_to_ij(env,s,sp)] -= 1 
				cell_assignments.append((agent,action))

			# else assign stay still 
			else:
				action = 0 
				cell_assignments.append((agent,action))
				round_errors += 1

		# else stay still  
		else:
			action = 0 
			cell_assignments.append((agent,action))
			round_errors += 1

	return cell_assignments