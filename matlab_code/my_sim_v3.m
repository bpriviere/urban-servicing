

function [sim_results] = my_sim_v3(param, initial_agents)


    % agents
    agents = initial_agents;
    
    if 1
        [Q_MDP, ~, ~, param] = solve_MDP(param, param.training_data);
        for i = 1:param.Ni
            agents(i).Q = Q_MDP;
        end
    end
    
    sim_results.Q = zeros( param.Nq, param.Ni, param.Nt); % track Q estimates
    sim_results.R = zeros( param.Nt, 1); % reward function of time
    sim_results.LOC = zeros( 2, param.Ni, param.Nt); % tracks agent location
    sim_results.ACT = zeros( param.Ni, param.Nt);
    sim_results.data = zeros( param.Ni * param.Nt, 7); % remembers request data 
    Delta_err = zeros(param.Nt,1);
    Delta_z = zeros(param.Nt, 1);
    refresh = zeros(param.Nt,1);
    for k = 1:param.Nt
        
        fprintf('    delta_err = %d\n', calc_delta_error(param,k))
        fprintf('    delta_des = %d\n', param.delta_des)
        
        if param.update_on
            if or( ...
                    and( param.k_on, k - param.last_update_k >= param.k), ...
                    and( ~param.k_on, calc_delta_error(param, k)> param.delta_des))
                fprintf('Update At k = %d\n', k)
                
                agents_data = clean_data(sim_results.data);
                [Q_MDP, ~, ~, param] = solve_MDP(param, agents_data);
                for i = 1:param.Ni
                    agents(i).Q = Q_MDP;
                end

                refresh(k) = 1;
%                 param = reset_error_bounds(param);
            end
        end

        [agents, requests] = get_local_info(param, agents, k);

        agents = TaskAssignment2(param, agents, requests);
        agents = CalculateUpdate(param, agents, requests);
        agents = CalculateReward(param, agents, requests);

        if k ~= param.last_update_k
            if param.algo == 0
                agents = CKIF(param, agents);
            elseif param.algo == 2
                agents = DIST_SS(param,agents);
            end
        end

        agents = moveAgents(param, agents, requests);
        sim_results = saveTimestep(param, requests, agents, sim_results, k);
        
        if ~param.k_on
            0;
%             param = calc_bounds(param, agents, requests);
        end
        
        if mod(k,5) == 0
            fprintf('Completed: %d/%d \n', k, param.Nt);
%             if and( param.algo == 1, param.plot_movie_on)
%                 makeMoviePlots(param, agents, requests, k);
%             end    
        end
    end
end