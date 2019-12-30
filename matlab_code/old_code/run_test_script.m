

% SCRIPT FOR NUMERICAL EXPERIMENTS

clear all; close all; clc;

rng(4);

param = my_set_param();
agents = init_agents(param);

param.n_episodes = 3;
R_c_sarsa_1 = zeros(param.Nt,param.n_episodes);
R_c_mpi_1 = zeros(param.Nt,param.n_episodes);
% centralized sarsa-1 (one iteration)
for i_episode = 1:param.n_episodes
    fprintf('Episode %d/%d\n', i_episode, param.n_episodes)
    
    param.algo = 0;
    param.mdp_on= 0 ;
    [sim_results] = my_sim_v3(param, agents);
    R_c_sarsa_1(:,i_episode) = sim_results.R;

    % centralized MPI
    param.algo = 0;
    param.mdp_on= 1;
    param.k = 1;
    [sim_results] = my_sim_v3(param, agents);
    R_c_mpi_1(:,i_episode) = sim_results.R;
    
end

figure()
h1 = plot( cumsum(mean(R_c_sarsa_1,2)),'DisplayName','C-SARSA-1');
h2 = plot( cumsum(mean(R_c_mpi_1,2)),'DisplayName','C-MPI-1');
legend('location','best')


