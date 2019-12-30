function agents = CalculateUpdate(param, agents, requests)

    for i = 1:param.Ni
        z = zeros(param.Nq,1);
        for s = 1:param.Ns
            a = requests(agents(i).curr_action_i);
            q_idx = sa_to_q(param, s, a);
            R_sa = calcReward(param, s, a);
            dropoff_s = loc_to_state(param, a.dropoff_x, ...
                a.dropoff_y);
            Qp_sa = max(extract_q_from_s(param, dropoff_s, agents(i).Q));
            z(q_idx) = R_sa + param.gamma * Qp_sa;
            z(q_idx) = z(q_idx) + param.MeasurementNoise*randn();
        end
        
        if ~param.fix_on
            for q = 1:param.Nq
                if z(q) == 0
                    z(q) = (1-param.delta_f)*agents(i).Q(q);
                end
            end
        end
        
        agents(i).z = z;
    end
    
end


