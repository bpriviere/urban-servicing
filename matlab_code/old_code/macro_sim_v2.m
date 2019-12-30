

clear all; close all; clc;

% MACRO SIM VERSION

rng(4);
param = my_set_param();
param.n_episodes = 10;

CASES_K = [1, 25, 50, 100];
CASES_Nx = [2, 3, 4, 5];

J_norm_mean = zeros(length(CASES_K), length(CASES_Nx), 2);
J_norm_std = zeros(length(CASES_K), length(CASES_Nx), 2);
R_case_mean = zeros(length(CASES_K), length(CASES_Nx), param.Nt, 3);
R_case_std = zeros(length(CASES_K), length(CASES_Nx), param.n_episodes, 3);

tic; 
count = 0;
for i_k = 1:length(CASES_K)
    for i_nx = 1:length(CASES_Nx)
        count = count + 1;
        fprintf('Case: %d/%d\n', count, length(CASES_K)*length(CASES_Nx))
        param.Nx = CASES_Nx(i_nx);
        param.Ns = param.Nx*param.Ny;
        param.Nq = param.Ns^3;

        agents = init_agents(param);
        generateData(param, param.input_data_filename);
        generateData(param, param.training_data_filename);
        param.data = dlmread(param.input_data_filename);
        param.training_data = dlmread(param.training_data_filename);  
        
        R_MPI = zeros( param.Nt, param.n_episodes);
        R_no_MPI = zeros( param.Nt, param.n_episodes);
        R_c = zeros( param.Nt, param.n_episodes);
        R_g = zeros( param.Nt, param.n_episodes);
        Q_MPI = zeros(param.Nq, param.Ni, param.Nt,  param.n_episodes);
        Q_no_MPI = zeros(param.Nq, param.Ni, param.Nt,  param.n_episodes);
        Q_c = zeros(param.Nq, param.Ni, param.Nt,  param.n_episodes);
        for i_ep = 1:param.n_episodes
            fprintf('Episode %d/%d \n', i_ep, param.n_episodes)

            param.algo = 2; 
            param.mdp_on = 1;
            param.k = CASES_K(i_k);
            fprintf('\t wMDP\n')
            [sim_results] = my_sim_v3(param, agents);
            R_MPI(:, i_ep) = sim_results.R;
            Q_MPI(:, :, :, i_ep) = sim_results.Q;
            LOC_MPI = sim_results.LOC;
            ACT = sim_results.ACT;

            param.mdp_on = 0;
            fprintf('\t woMDP\n')
            [sim_results] = my_sim_v3(param, agents);
            R_no_MPI(:, i_ep) = sim_results.R;
            Q_no_MPI(:, :, :, i_ep) = sim_results.Q;

            param.algo = 0;
            param.k = 1;         
            param.mdp_on = 1;
            fprintf('\t Centralized w/Full MPI \n')
            [sim_results] = my_sim_v3(param, agents);
            R_c(:, i_ep) = sim_results.R;
            Q_c(:, :, :, i_ep) = sim_results.Q;
                

        %     fprintf('\t Greedy\n')
        %     [sim_results] = greedy_ver(param);
        %     R_g(:, i_ep) = sim_results.R;
        end
        R_case_mean(i_k, i_nx, :,1) = mean(R_MPI,2);
        R_case_std(i_k, i_nx, :,1) = std(R_MPI);
        
        R_case_mean(i_k, i_nx, :,2) = mean(R_no_MPI,2);
        R_case_mean(i_k, i_nx, :,3) = mean(R_c,2);
        R_case_std(i_k, i_nx, :,2) = std(R_no_MPI);
        R_case_std(i_k, i_nx, :,3) = std(R_c);
        
        J_norm_mpi = sum(R_MPI)./sum(R_c);
        J_norm_no_mpi = sum(R_no_MPI)./sum(R_c);
        
        
        J_norm_mean(i_k, i_nx, 1) = mean(J_norm_mpi);
        J_norm_mean(i_k, i_nx, 2) = mean(J_norm_no_mpi);
        J_norm_std(i_k, i_nx, 1) = std(J_norm_mpi);
        J_norm_std(i_k, i_nx, 2) = std(J_norm_no_mpi);
        
    end
end
sim_runtime = toc;
fprintf('Runtime: %d\n', sim_runtime);

% MACRO 
figure()
xlabel('Nx'), ylabel('Normalized J')
for i_k = 1:length(CASES_K)
    plot(CASES_Nx, J_norm_mean(i_k,:,1),'s-',...
        'DisplayName', sprintf('k = %d', CASES_K(i_k)),...
        'MarkerSize',param.markersize,'Linewidth',param.linewidth);
end
plot(CASES_Nx, J_norm_mean(i_k,:,2),'^-',...
    'DisplayName', 'SARSA-D', 'MarkerSize',param.markersize,'Linewidth',param.linewidth);
grid on, legend('location','best')
set(gca,'FontSize',param.fontsize)

% State Space
plot_state_space_v2(param, LOC_MPI, param.Nt);

% action frequency
% plot_action_freq(param, ACT);

% tracking
% plot_micro_q_v2(param, mean(Q_MPI,4), mean(Q_no_MPI,4), mean(Q_c,4));
% plot_top_policies(param, mean(Q_MPI,4), mean(Q_no_MPI,4), mean(Q_c,4));
% plot_testing_data(param, param.training_data);
% plot_testing_data(param, param.data);

% performance
% figure()
% subplot(2,1,1)
% plot(mean(R_MPI,2))
% plot(mean(R_no_MPI,2))
% plot(mean(R_c,2))
% % plot(mean(R_g,2))
% subplot(2,1,2)
% plot(cumsum(mean(R_MPI,2)))
% plot(cumsum(mean(R_no_MPI,2)))
% plot(cumsum(mean(R_c,2)))
% % plot(cumsum(mean(R_g,2)))
% legend('MPI','No MPI','Central','Greedy','location','best');
