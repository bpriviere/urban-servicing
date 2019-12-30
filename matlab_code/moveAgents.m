function agents = moveAgents(param, agents, requests)

    for i = 1:param.Ni
        agents(i).x = requests(agents(i).curr_action_i).dropoff_x;
        agents(i).y = requests(agents(i).curr_action_i).dropoff_y;
    end
end