function [agents] = UpdateAgentsDist(param, agents, requests)

    % clear previous requests
    for i = 1:param.Ni
        agents(i).requests = struct();
        agents(i).nr = 0;
        agents(i).neighbors = struct();
        agents(i).nn = 0;
    end

    % get local requests
    for i = 1:param.Ni
        for j = 1:param.Nr
            r_x = requests(j).pickup_x;
            r_y = requests(j).pickup_y;
            dist = norm( [ r_x, r_y] - [agents(i).x, agents(i).y] ); % degree lat long
            if dist < param.R_task/param.scale
                agents = add_Request_j_to_Agent_i(i,j,requests,agents);
            end
        end
    end

    % get neighbors
    for i = 1:param.Ni
        for j = 1:param.Ni
            dist = norm( [agents(i).x agents(i).y] - [agents(j).x agents(j).y]);
            if dist < param.R_comm/param.scale
                agents = add_Agent_j_to_Agent_i(i,j,agents);
            end
        end
    end
end