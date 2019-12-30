
function [agents, sim_results] = main_opt_routing(param, agents, Q_MDP)

    % initialize
%     init_sim; 
    for i_agent = 1:param.Ni
        agents(i_agent).Q = Q_MDP;
    end

    % timeloop
    sim_results.Q = zeros( param.Nq, param.Ni, param.Nt); % track Q estimates
    sim_results.R = zeros( param.Nt, 1); % reward function of time
    sim_results.A = zeros( 2, param.Ni, param.Nt); % tracks agent location
    sim_results.data = zeros( param.Ni * param.Nt, 6); % remembers request data 
    for k = 1:param.Nt

        requests = getRequests(param,k);

        if param.algo == 0
            agents = UpdateAgentsCentralized(param, agents, requests);
        elseif or(param.algo == 1, param.algo == 2)
            agents = UpdateAgentsDist(param, agents, requests);
        end
        [agents, requests] = add_null_actions(param, agents, requests);

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

        if and( param.algo == 1, param.plot_movie_on)
            makeMoviePlots(param, agents, requests, k);
        end    

        agents = moveAgents(param, agents, requests);

        sim_results = saveTimestep(param, requests, agents, sim_results, k);
        if mod(k,10) == 0
            fprintf('Completed: %d/%d \n', k, param.Nt);
%             plot_state_space(param, agents,k)
        end
    end
end


% plot_q_c(param, sim_results);
% plot_r(param, sim_results);
% plot_a(param, sim_results);