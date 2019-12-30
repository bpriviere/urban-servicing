function [agents, requests] = get_local_info(param, agents, k)

    requests = getRequests(param,k);

    if param.algo == 0
        agents = UpdateAgentsCentralized(param, agents, requests);
    elseif or(param.algo == 1, param.algo == 2)
        agents = UpdateAgentsDist(param, agents, requests);
    end
    [agents, requests] = add_null_actions(param, agents, requests);

end