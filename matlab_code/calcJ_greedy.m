function J = calcJ_greedy(param, agents, requests, i, action_i)

    used_requests = zeros(param.Ni,1);
%     used_requests = zeros(param.Ni);
    for j = 1:param.Ni
        if j ~= i
            used_requests(j) = agents(j).curr_action_i;
        end
    end

    if any( action_i == used_requests)
        J = -3;
    else
        a = requests(action_i);
        s = loc_to_state(param, agents(i).x, agents(i).y);
        tts = calcTTS(param,s,a);
        J =  a.revenue - param.C1*(tts + a.ttc) + param.MeasurementNoise*randn;
%         J = requests(action_i).revenue;
    end
end