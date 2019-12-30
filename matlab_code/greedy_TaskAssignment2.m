function agents = greedy_TaskAssignment2(param, agents, requests)

    % Distributed Task Assignment, Binary Log-linear Learning, Game Theory
    same_action_count = zeros(param.Ni, 1);

    % randomly assign a task
    for i = 1:param.Ni
        % if this agent has available requests, randomly initialize to one
        if agents(i).nr > 1
            agents(i).curr_action_i = agents(i).requests(...
                randi([1,agents(i).nr],1,1)).i;
        else % if no available requests,  choose null action and mark as converged
            agents(i).curr_action_i = agents(i).requests(1).i;
            same_action_count(i) = param.task_convergence;
        end
    end

    n_trial_task_assignment = 0;
    % all agents have to converge
    while any(same_action_count < param.task_convergence)
        n_trial_task_assignment = n_trial_task_assignment + 1;

        % choose a random agent that has not converged
        nc_agents_i = find(same_action_count < param.task_convergence);
        i = nc_agents_i(randi([1,length(nc_agents_i)],1,1));
        
        % new assignment proposed
        agents(i).prop_action_i = agents(i).requests(randi(...
            agents(i).nr)).i;
        while agents(i).prop_action_i == agents(i).curr_action_i
            agents(i).prop_action_i = agents(i).requests(randi(...
                agents(i).nr)).i;
        end
        
        J = calcJ_greedy(param, agents, requests, i, agents(i).curr_action_i);
        J_p = calcJ_greedy(param, agents, requests, i, agents(i).prop_action_i); 

        tau = param.tau;
        P_i = exp(J/tau)/(exp(J/tau) + exp(J_p/tau));

        % if J >> J_p -> P_i = 1, rand < P_i, (you want to stay at that action)
        % if J_p >> J -> P_i = 0, rand > P_i, (you want to leave that action) 
        if rand > P_i
            agents(i).curr_action_i = agents(i).prop_action_i;
            same_action_count(i) = 0;
        else
            same_action_count(i) = ...
                same_action_count(i) + 1;
        end 
    end

    for i = 1:param.Ni
        for j = 1:param.Ni
            if and(agents(i).curr_action_i == agents(j).curr_action_i, i ~= j)
                'help'
            end
        end
    end


end