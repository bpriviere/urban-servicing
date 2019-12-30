
function [sim_results] = greedy_ver(param, initial_agents)

    % initialize
    agents = initial_agents;

    % timeloop
    sim_results.Q = zeros( param.Nq, param.Ni, param.Nt); % track Q estimates
    sim_results.R = zeros( param.Nt, 1); % reward function of time
    sim_results.A = zeros( 2, param.Ni, param.Nt); % tracks agent location
    for k = 1:param.Nt

        requests = getRequests(param,k);

        agents = UpdateAgentsDist(param, agents, requests);
        [agents, requests] = add_null_actions(param, agents, requests);

        agents = greedy_TaskAssignment2(param, agents, requests);
        agents = CalculateReward(param, agents, requests);

        agents = moveAgents(param, agents, requests);    


        reward_k = 0;
        for i = 1:param.Ni
            sim_results.A(1:2,i,k) = [agents(i).x, agents(i).y];
            reward_k = reward_k + agents(i).reward;
        end
        sim_results.R(k) = reward_k;

        if mod(k,5) == 0
            fprintf('Completed: %d/%d \n', k, param.Nt);
        end

    %     if param.plot_movie_on
    %         makeMoviePlots(param, agents, requests, k);
    %     end    

    end

end


% plot_q_c(param, sim_results);
% plot_r(param, sim_results);
% plot_a(param, sim_results);