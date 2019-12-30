
clear all; close all; clc;

% param
param = my_set_param();
param.algo = 2;
param.n_episodes = 10;

R_MPI = zeros( param.Nt, 1);
R_no_MPI = zeros( param.Nt, 1);

for i_ep = 1:param.n_episodes
    % agents
    agents = init_agents(param);

    fprintf('Solving Offline MDP\n')
    [Q_MDP, P_MDP, R_MDP, param] = solve_MDP(param, param.training_data);
    fprintf('Solved Offline MDP\n')
    for i = 1:param.Ni
        agents(i).Q = Q_MDP;
    end

    sim_results.Q = zeros( param.Nq, param.Ni, param.Nt); % track Q estimates
    sim_results.R = zeros( param.Nt, 1); % reward function of time
    sim_results.A = zeros( 2, param.Ni, param.Nt); % tracks agent location
    sim_results.data = zeros( param.Ni * param.Nt, 6); % remembers request data 
    Delta_err = zeros(param.Nt,1);
    Delta_z = zeros(param.Nt, 1);
    refresh = zeros(param.Nt,1);
    for k = 1:param.Nt

        Delta_err(k) = calc_delta_error(param, k);
        if Delta_err(k) > param.delta_des
            fprintf('Request Sent to Centralized\n')
            agents_data = clean_data(sim_results.data);
            [Q_MDP, P_MDP, R_MDP, param] = solve_MDP(param, [param.training_data; agents_data]);
            param.last_update_k = k;
            for i = 1:param.Ni
                agents(i).Q = Q_MDP;
            end
            refresh(k) = 1;
        end

        [agents, requests] = get_local_info(param, agents, k);

        agents = TaskAssignment2(param, agents, requests);
        agents = CalculateUpdate(param, agents, requests);
        agents = CalculateReward(param, agents, requests);

        delta_z_max = 0;
        for i = 1:param.Ni
            if any(agents(i).z > delta_z_max)
                delta_z_max = max(agents(i).z);
            end
        end
        Delta_z(k) = delta_z_max;
        param.delta_z_max = max(Delta_z);

        if param.algo == 0
            agents = CKIF(param, agents);
        elseif param.algo == 1
            agents = DIST_MS(param, agents);
        elseif param.algo == 2
            agents = DIST_SS(param,agents);
        end

        for i = 1:param.Ni
            if param.underbar_alpha < agents(i).alpha
                param.underbar_alpha = agents(i).alpha;
            end
        end

        agents = moveAgents(param, agents, requests);

        sim_results = saveTimestep(param, requests, agents, sim_results, k);

        if mod(k,2) == 0
            fprintf('Completed: %d/%d \n', k, param.Nt);
            if and( param.algo == 1, param.plot_movie_on)
                makeMoviePlots(param, agents, requests, k);
            end    
        end
    end
    R_MPI = R_MPI + sim_results.R;
    if i_ep == 1
        plot_state_space(param, agents, k)
    end


    sim_results.Q = zeros( param.Nq, param.Ni, param.Nt); % track Q estimates
    sim_results.R = zeros( param.Nt, 1); % reward function of time
    sim_results.A = zeros( 2, param.Ni, param.Nt); % tracks agent location
    sim_results.data = zeros( param.Ni * param.Nt, 6); % remembers request data 
    agents = init_agents(param);
    for k = 1:param.Nt

        [agents, requests] = get_local_info(param, agents, k);

        agents = TaskAssignment2(param, agents, requests);
        agents = CalculateUpdate(param, agents, requests);
        agents = CalculateReward(param, agents, requests);

        if param.algo == 0
            agents = CKIF(param, agents);
        elseif param.algo == 1
            agents = DIST_MS(param, agents);
        elseif param.algo == 2
            agents = DIST_SS(param,agents);
        end

        agents = moveAgents(param, agents, requests);
        sim_results = saveTimestep(param, requests, agents, sim_results, k);
    end
    R_no_MPI = R_no_MPI + sim_results.R;
end

R_MPI = R_MPI / param.n_episodes;
R_no_MPI = R_no_MPI / param.n_episodes;


figure()
subplot(2,1,1);
plot(R_no_MPI, 'DisplayName','No MPI');
plot(R_MPI, 'DisplayName','MPI');
subplot(2,1,2);
plot(cumsum(R_no_MPI), 'DisplayName','No MPI');
plot(cumsum(R_MPI), 'DisplayName','MPI');
legend('location','best')

figure()
subplot(2,1,1)
plot(Delta_err); grid on
title('\delta_{err}')
subplot(2,1,2)
plot(Delta_z); grid on
title('\delta_z')

