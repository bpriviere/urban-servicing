
function sim_results = saveTimestep(param, requests, agents, sim_results, k)

    reward_k = 0;
    request_data = zeros(param.Ni,7);
    for i = 1:param.Ni
        sim_results.Q(:, i, k) = agents(i).Q;
        sim_results.LOC(1:2,i,k) = [agents(i).x, agents(i).y];
        
        s_curr = loc_to_state(param, agents(i).x, agents(i).y);
        sim_results.ACT(i,k) = sa_to_q(param, s_curr, requests(agents(i).curr_action_i));
        
        request = requests(agents(i).curr_action_i);
        request_data(i,:) = [request.pickup_x, request.pickup_y, request.dropoff_x, ...
            request.dropoff_y, request.revenue, request.ttc, request.t];
        reward_k = reward_k + agents(i).reward;
    end 
    sim_results.R(k) = reward_k;
    idxs = ((k-1)*param.Ni + 1) : ( k*param.Ni);
    sim_results.data(idxs,:) = request_data;
end
