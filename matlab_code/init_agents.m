function agents = init_agents(param)


% initialize agents
agents = {};
for i = 1:param.Ni

    if param.real_data_on
        s0 = randi(param.Ns);
        map_s0 = param.my_state_to_map_state(s0);
        [agents(i).x, agents(i).y] = point_in_poly( param.map(map_s0).BoundingBox, ...
            param.map(map_s0).X, param.map(map_s0).Y);
        
    else
        agents(i).x = param.Nx*rand();
        agents(i).y = param.Ny*rand(); 
    end
    
    agents(i).P = param.P0*eye(param.Nq);
    agents(i).i = i;
    agents(i).Q = param.Q0*ones(param.Nq,1);
    agents(i).alpha = ones(param.Nq);
end


end