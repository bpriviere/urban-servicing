

clear all; close all; 
clc;

% rng(4);

my_set_param;
agents = initialize_agents(param);

param.n_episodes = 3;
param.algo = 1;
stacked_results = {};
for i_episode = 1:param.n_episodes
    if i_episode == 1
        Q_MDP = solve_MDP(param, param.training_data);
    else
        Q_MDP = solve_MDP(param, sim_results.data);
    end
    [agents, sim_results] = main_opt_routing(param, agents, Q_MDP);
    stacked_results = stack_results( param, sim_results, stacked_results);
end
R_MPI = stacked_results.R;

stacked_results = {};
for i_episode = 1:param.n_episodes
    Q_MDP = param.Q0*ones(param.Nq,1);
    [agents, sim_results] = main_opt_routing(param, agents, Q_MDP);
    stacked_results = stack_results( param, sim_results, stacked_results);
end
R_no_MPI = stacked_results.R;

figure()
plot(cumsum(R_MPI),'DisplayName','MPI');
plot(cumsum(R_no_MPI),'DisplayName','No MPI');
legend('location','best')

writeResults(param, stacked_results);
plot_state_space(param, agents, 1);
plot_macro_r(param);
% plot_qbar(param);
plot_micro_q(param);