function agents = TaskAssignment(param, agents, requests)

    % temp assign tasks based on distance
    dist = Inf(param.Ni, param.Nr);
    for i = 1:param.Ni
        agent_x = agents(i).x;
        agent_y = agents(i).y;
        for j = 1:agents(i).nr
            request = requests(agents(i).requests(j).i);
            if request.revenue ~= 0
                request_x = request.pickup_x;
                request_y = request.pickup_y;
                dist(i,j) = norm( [agent_x, agent_y] - [request_x,request_y]);
            end
        end
    end

    used_requests = zeros(param.Ni,1);
    for i = 1:param.Ni
        [~,inds] = sort(dist(i,:));
        count = 1;
        agents(i).curr_action_i = agents(i).requests(inds(count)).i;
        while any( agents(i).requests(inds(count)).i == used_requests)
            count = count + 1;
            agents(i).curr_action_i = agents(i).requests(inds(count)).i;
        end
        used_requests(i) = agents(i).requests(inds(count)).i;
    end
end

