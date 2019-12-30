

function [agents] = UpdateAgentsCentralized(param, agents, requests)

    % clear previous requests
    for i = 1:param.Ni
        agents(i).requests = struct();
        agents(i).nr = 0;
    end

    % add all new requests
    for i = 1:param.Ni
        for j = 1:param.Nr
            agents = add_Request_j_to_Agent_i(i,j,requests,agents);
        end
    end
    
    
    
end