function q_idx = sa_to_q(param,s,a)
    try
        s_p = loc_to_state(param, a.pickup_x, a.pickup_y);
        s_d = loc_to_state(param, a.dropoff_x, a.dropoff_y);
        q_idx = (s-1)*param.Ns*param.Ns + (s_p-1)*param.Ns + s_d;
    catch
        q_idx = 0; % null action 
    end
end