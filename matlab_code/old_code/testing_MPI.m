
% testing MPI

close all; clear all; clc;

my_set_param;

P_MDP = get_MDP_P(param, param.data);
R_MDP = get_MDP_R(param, param.data);

[V, policy, iter, cpu_time] = ...
    mdp_policy_iteration_modified(P_MDP, R_MDP, param.gamma);

Q_MDP = get_MDP_Q(param, V, R_MDP);

Q_MDP_T = Q_MDP';
Q_MDP = Q_MDP_T(:);

% mdp_check(P_MDP , R_MDP)