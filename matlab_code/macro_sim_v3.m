

clear all; close all; clc;

% MACRO SIM VERSION 3

rng(777);
hold all; 

param = my_set_param();

J_mpi = zeros( length(param.CASES_k), length(param.CASES_Nt), ...
    length(param.CASES_Ni), param.n_episodes);
J_greedy = zeros( size(J_mpi));
J_c_closest = zeros( size(J_mpi));
J_d_closest = zeros( size(J_mpi));
J_c_woupdate = zeros( size(J_mpi));
J_c_wupdate = zeros( size(J_mpi));
J_d_woupdate = zeros( size(J_mpi));
J_d_wupdate = zeros( size(J_mpi));
time_of_sim = zeros( horzcat(size(J_mpi),6));

count = 0;
n_case = length(param.CASES_k)*length(param.CASES_Nt)*length(param.CASES_Ni);
for i_i = 1:length(param.CASES_Ni)
    param.Ni = param.CASES_Ni(i_i);
    param.Nr = param.Ni;
%     param.Nr = 2*param.Ni;
    
    for i_k = 1:length(param.CASES_k)
        for i_nt = 1:length(param.CASES_Nt)
            count = count + 1;
            fprintf('Case: %d/%d\n', count, n_case)

            param.Nt = param.CASES_Nt(i_nt);

            R_mpi = zeros(param.Nt, param.n_episodes);
            R_greedy = zeros(param.Nt, param.n_episodes);
            R_c_closest = zeros(param.Nt, param.n_episodes);        
            R_d_closest = zeros(param.Nt, param.n_episodes);
            R_c_wupdate  = zeros(param.Nt, param.n_episodes);
            R_d_wupdate  = zeros(param.Nt, param.n_episodes); 
            R_d_woupdate = zeros(param.Nt, param.n_episodes); 
            R_c_woupdate = zeros(param.Nt, param.n_episodes); 

            Q_mpi = zeros( param.Nq, param.Ni, param.Nt, param.n_episodes);
            Q_c_wupdate  = zeros( param.Nq, param.Ni, param.Nt, param.n_episodes);
            Q_d_wupdate  = zeros( param.Nq, param.Ni, param.Nt, param.n_episodes);
            Q_d_woupdate = zeros( param.Nq, param.Ni, param.Nt, param.n_episodes);
            Q_c_woupdate = zeros( param.Nq, param.Ni, param.Nt, param.n_episodes);

            for i_ep = 1:param.n_episodes
                fprintf('\tCase: %d/%d, Episode %d/%d \n', count, n_case, i_ep, param.n_episodes)

                if param.real_data_on
                    fprintf('Processing City Data\n')
                    [param, param.data] = readCityData_v2(param, param.input_data_filename);
                    param.data = remove_outliers(param, param.data);
                    param.training_data = readTrainingData(param, param.training_data_filename);
                    param.training_data = remove_outliers(param, param.training_data);
                else
                    param.data = generateData(param, param.input_data_filename);
                    param.training_data = generateData(param, param.training_data_filename); 
                end

                agents = init_agents(param);

    %             MPI soln
%                 param.algo = 0; 
%                 param.update_on = 1;
%                 param.k = 1;
%                 param.k_on = 1;
%                 fprintf('\t\t MPI\n')
%                 tic;
%                 [sim_results] = my_sim_v3(param, agents);
%                 time_of_sim(i_k, i_nt, i_i, i_ep, 1) = toc;
%                 R_mpi( :, i_ep) = sim_results.R;
%                 Q_mpi( :, :, :, i_ep) = sim_results.Q;
%                 J_mpi(i_k, i_nt, i_i, i_ep) = sum( R_mpi(:,i_ep));
%                 LOC_MPI = sim_results.LOC;

                % Distributed, with mpi
                if param.macro_k_on
                    param.k_on = 1;
                else
                    param.k_on = 0;
                end
                param.algo = 2; 
                param.update_on = 1;
                param.k = param.CASES_k(i_k);
                fprintf('\t\t D wMPI\n')
                tic;
                [sim_results] = my_sim_v3(param, agents);
                time_of_sim(i_k, i_nt, i_i, i_ep, 4) = toc;
                loc_d_wmpi = sim_results.LOC;
                R_d_wupdate( :, i_ep) = sim_results.R;
                Q_d_wupdate( :, :, :, i_ep) = sim_results.Q;
                J_d_wupdate( i_k, i_nt, i_i, i_ep) = sum( R_d_wupdate(:,i_ep));

                % Distributed, without mpi
                param.update_on = 0;
                param.algo = 2;
                fprintf('\t\t D woMPI\n')
                tic;
                [sim_results] = my_sim_v3(param, agents);
                time_of_sim(i_k, i_nt, i_i, i_ep, 5) = toc;
                R_d_woupdate( :, i_ep) = sim_results.R;
                Q_d_woupdate( :, :, :, i_ep) = sim_results.Q;
                J_d_woupdate( i_k, i_nt, i_i, i_ep) = sum(R_d_woupdate(:,i_ep));

                % Closest Policy
                param.algo = 2;
                fprintf('\t\t Closest Distributed\n')
                tic;
                [sim_results] = closest_sim(param, agents);
                time_of_sim(i_k, i_nt, i_i, i_ep, 6) = toc;
                R_d_closest(:, i_ep) = sim_results.R;
                J_d_closest(i_k, i_nt, i_i, i_ep) = sum(R_d_closest(:,i_ep));
            end
        end
    end
end

sim_runtime = sum(sum(sum(sum(time_of_sim))));
save('sim_workspace')
fprintf('Runtime: %d\n', sim_runtime);

% Performance plots
% Over Time
plot_agent_idx = size(param.CASES_Ni,1);
plot_performance_nt( param, param.CASES_Nt, param.CASES_k, plot_agent_idx, J_d_wupdate, ...
    J_d_woupdate, J_d_closest, J_mpi); 
title(sprintf('Performance for %d Taxis', param.CASES_Ni(plot_agent_idx)))
plot_time_idx = size(param.CASES_Nt,1);
plot_performance_ni( param, param.CASES_Ni, param.CASES_k, plot_time_idx, J_d_wupdate, ...
    J_d_woupdate, J_d_closest, J_mpi); 
title(sprintf('Performance for %d Timesteps', param.CASES_Nt(plot_time_idx)))

if param.real_data_on

    data_idx = 1:(param.Nt*param.Nr);
    time_idx = 1:param.Nr:(param.Nt*param.Nr) ;
    days = param.data(data_idx, 7);
    hours_interp = (interp1(data_idx, days, time_idx)-1)*24;

    figure()
    hold on
    plot(hours_interp, cumsum(mean(R_mpi,2)),'linewidth',param.linewidth)
    plot(hours_interp, cumsum(mean(R_d_wupdate,2)),'linewidth',param.linewidth)
    plot(hours_interp, cumsum(mean(R_d_woupdate,2)),'linewidth',param.linewidth)
    plot(hours_interp, cumsum(mean(R_d_closest,2)),'linewidth',param.linewidth)
    legend('MPI','D: wUpdate','D: woUpdate','Closest','location','best');
%     legend('D: wUpdate','D: woUpdate','Closest','location','best');
    xlabel('Hours of May 1st, 2017'), ylabel('USD'), 
    % set(gca,'XTick',1:23)
    set(gca,'FontSize',20)
    grid on
end

% STATE SPACE
% plot_state_space_v2(param, LOC_MPI, 1);


% Q TRACE
% err_q_c = abs(Q_c_wupdate - Q_c_woupdate);
% err_q_d_wupdate = abs(Q_c_wupdate - Q_d_wupdate);
% err_q_d_woupdate = abs(Q_c_wupdate - Q_d_woupdate);
% plot_micro_qerror_v3(param, err_q_c, err_q_d_wupdate, err_q_d_woupdate)
% plot_qerror_v3_sa(param, err_q_c, err_q_d_wupdate, err_q_d_woupdate, 1,2,2)
% plot_micro_q_v3(param, Q_c_wupdate, Q_c_woupdate, Q_d_wupdate, Q_d_woupdate)

figure()
plot(cumsum(mean(R_mpi,2)),'linewidth',param.linewidth)
% plot(cumsum(mean(R_c_wupdate,2)),'linewidth',param.linewidth)
% plot(cumsum(mean(R_c_woupdate,2)),'linewidth',param.linewidth)
plot(cumsum(mean(R_d_wupdate,2)),'linewidth',param.linewidth)
plot(cumsum(mean(R_d_woupdate,2)),'linewidth',param.linewidth)
% plot(cumsum(mean(R_c_closest,2)),'linewidth',param.linewidth)
plot(cumsum(mean(R_d_closest,2)),'linewidth',param.linewidth)
legend('MPI', 'D: wUpdate',...
    'D: woUpdate','D: Closest','location','best');
% legend('MPI', 'C: wUpdate','C: woUpdate','D: wUpdate',...
%     'D: woUpdate','C: Closest','D: Closest','location','best');
set(gca,'FontSize',20)
grid on

% ACTION FREQ
% plot_action_freq(param, ACT);


    %             % centralized, with mpi update
    %             param.algo = 0; 
    %             param.update_on = 1;
    %             param.k = param.CASES_k(i_k); 
    %             fprintf('\t\t C wMPI\n')
    %             tic;
    %             [sim_results] = my_sim_v3(param, agents);
    %             time_of_sim(i_k, i_nt, i_i, i_ep, 2) = toc;
    %             R_c_wupdate( :, i_ep) = sim_results.R;
    %             Q_c_wupdate( :, :, :, i_ep) = sim_results.Q;
    %             J_c_wupdate(i_k,i_nt,i_ep) = sum( R_c_wupdate(:,i_ep));
    %             
    %             % centralized, without mpi
    %             param.update_on = 0;
    %             param.algo = 0;
    %             fprintf('\t\t C woMPI\n')
    %             tic;
    %             [sim_results] = my_sim_v3(param, agents);
    %             time_of_sim(i_k, i_nt, i_i, i_ep, 3) = toc;
    %             R_c_woupdate( :, i_ep) = sim_results.R;
    %             Q_c_woupdate( :, :, :, i_ep) = sim_results.Q;
    %             J_c_woupdate(i_k,i_nt,i_ep) = sum( R_c_woupdate(:,i_ep));
