

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

	cell_assignments = []

	if len(agents)>0:

		# 
		nfree = len(agents)
		
		# reward is in [nq x 1]
		r = agents[0].r
		q = agents[0].q

		# normalized prediction in [ns x 1]
		w = np.zeros((env.param.env_ncell,env.param.rhc_horizon))
		for s in range(env.param.env_ncell):

			# q_idx = env.sa_to_q_idx(s,0)
			# w[s,:] = r[q_idx]

			pi = env.global_boltzmann_policy(q)
			w[s,:] = pi[s]

		for t in range(env.param.rhc_horizon):
			w[:,t] = w[:,t] / sum(w[:,t])

		# initial condition
		x0 = np.zeros((env.param.env_ncell))
		for agent in agents:
			s = env.coordinate_to_cell_index(agent.x,agent.y)
			x0[s] += 1

		# some preliminary manipulation 
		banned_idx = np.zeros((env.param.env_ncell,env.param.env_ncell**2))
		idxs_from_s = np.zeros((env.param.env_ncell,env.param.env_ncell**2))
		idxs_to_s = np.zeros((env.param.env_ncell,env.param.env_ncell**2))
		
		for i in range(env.param.env_ncell):
			for j in range(env.param.env_ncell):
				idxs_from_s[i,i_j_to_ij(env,i,j)] = 1 
				idxs_to_s[i,i_j_to_ij(env,j,i)] = 1 
				banned_idx[i,i_j_to_ij(env,i,j)] = 1

			local_states,local_actions = env.get_local_transitions(i)
			for j in local_states:
				banned_idx[i, i_j_to_ij(env,i,j)] = 0

		# decision variables 
		# 	x_t is number of taxis in each cell 
		# 	U_t[i,j] is defined as cells in i going to cell j at time t
		# 	U_t is flattened into array U_t[i*env_ncell+j,:] bc cvxpy doesnt accept > 2d

		# debug where each agent takes action 0 
		dbg = False
		if dbg: 
			x_t = np.zeros((env.param.env_ncell,env.param.rhc_horizon+1))
			U_t = np.zeros((env.param.env_ncell**2,env.param.rhc_horizon))

			for agent in agents:
				s = env.coordinate_to_cell_index(agent.x,agent.y)

				U_t[i_j_to_ij(env,s,s),:] += 1

			for t in range(env.param.rhc_horizon):
				x_t[:,t] = x0
		else:
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
		if dbg:
			constr.append(np.sum(U_t,axis=0)==nfree)
		else:
			constr.append(cp.sum(U_t,axis=0)==nfree)

		for t in range(env.param.rhc_horizon):

			# conservation 1: taxis in 's' at next timestep = taxis curr in 's' + taxis going into cell 's'
			constr.append(idxs_to_s @ U_t[:,t] == x_t[:,t+1])

			# conservation 2: taxis leaving 's' less than or equal to taxis curr in 's' 
			constr.append(idxs_from_s @ U_t[:,t] == x_t[:,t])
			# constr.append(idxs_to_s @ U_t[:,t] == x_t[:,t])

			# dynamics: travel between banned indices are not allowed 
			constr += [banned_idx @ U_t[:,t] == 0] 

			# cost fnc:
			if dbg:
				cost += env.param.mdp_gamma**t * np.sum(np.abs(x_t[:,t]/nfree - w[:,t]))
			else:
				# cost += env.param.mdp_gamma**t * cp.sum(cp.abs(x_t[:,t]/nfree - w[:,t]))
				cost += env.param.mdp_gamma**t * cp.sum_squares(x_t[:,t]/nfree - w[:,t])


		if dbg:

			print('env.param.env_nx: ',env.param.env_nx)
			print('env.param.env_ny: ',env.param.env_ny)
			print('env.param.env_ncell: ',env.param.env_ncell)
			for s in range(env.param.env_ncell):
				print('s: ',s)
				print('   banned_idx[s,:]: ',banned_idx[s,:])
				print('   idxs_from_s[s,:]: ',idxs_from_s[s,:])
				print('   idxs_to_s[s,:]: ',idxs_to_s[s,:])
			print('x0: ',x0)
			print('U_t:',U_t)
			print(constr)

			exit()

		obj = cp.Minimize(cost)
		prob = cp.Problem(obj, constr)
		# prob.solve(verbose=True, solver=cp.GUROBI)
		prob.solve(verbose=False, solver=cp.GUROBI)

		# needs to be ncells**2 x 1 
		cell_movement = U_t.value[:,0]

		# print('x_t.value[:,0]:', x_t.value[:,0])
		# print('w[:,0]:',w[:,0])
		# print('x_t.value[:,1]:', x_t.value[:,1])
		# print('w[:,1]:',w[:,1])

		# for t in range(env.param.rhc_horizon):
		# 	print('t:',t)
		# 	print('np.linalg.norm( w[:,t] - x_t.value[:,t]): ',(np.linalg.norm( w[:,t] - x_t.value[:,t])))
		# exit()

		# convert to integers
		cell_movement = np.round(cell_movement)	
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

		actions = [a for (_,a) in cell_assignments]
		print('actions:',actions)
			
	return cell_assignments