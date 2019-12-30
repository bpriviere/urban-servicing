function J = calcJ(param, agents, requests, i, action_i)
    
    used_requests = zeros(param.Ni,1);
    for j = 1:param.Ni
        if j ~= i
            used_requests(j) = agents(j).curr_action_i;
        end
    end

    if any( action_i == used_requests)
        J = -3;
    elseif requests(action_i).revenue == 0
        J = 0;
%         s = loc_to_state(param, requests(action_i).pickup_x, requests(action_i).pickup_y);
%         Qp = extract_q_from_s(param, s, agents(i).Q);
%         J = param.gamma*max(Qp);
    else
        s = loc_to_state(param, agents(i).x, agents(i).y);
        J = agents(i).Q( sa_to_q(param, s, requests(action_i)));
    end
    
end