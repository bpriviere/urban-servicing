

clear all; close all; 
clc;

% rng(4);

my_set_param;
% param.Ns

if param.cent_on
    param.algo = 0;
    fprintf('Centralized\n')
    [agents, sim_results] = main_opt_routing(param);
    writeResults(param, sim_results);
end

if param.greedy_on
    fprintf('Greedy\n')
    [agents, sim_results] = greedy_ver(param);
    writeResults_greedy(param, sim_results);
end

if param.dist_ms_on
    param.algo = 1;
    fprintf('DIST MS\n')
    [agents, sim_results] = main_opt_routing(param);
    writeResults(param, sim_results);
end

if param.dist_ss_on
    param.algo = 2;
    fprintf('DIST SS\n')
    [agents, sim_results] = main_opt_routing(param);
    writeResults(param, sim_results);
end

plot_state_space(param, agents, 1);
plot_macro_r(param);
% plot_qbar(param)
% plot_micro_q(param);


