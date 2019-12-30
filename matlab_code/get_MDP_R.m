function R = get_MDP_R(param, data)

    ntrips = size(data,1);
    R = zeros(param.Ns, param.Ns, param.Ns*param.Ns);
    normalizing_denom = zeros(param.Ns, param.Ns, param.Ns*param.Ns);
%     num_requests = zeros(param.Ns, param.Ns, param.Ns*param.Ns);
    
    tf = max(data(:,7)); 

    for i_trip = 1:ntrips

        a.pickup_x = data(i_trip,1);
        a.pickup_y = data(i_trip,2);
        a.dropoff_x = data(i_trip,3);
        a.dropoff_y = data(i_trip,4);       
        rev = data(i_trip,5);
        ttc = data(i_trip,6);
        tp = data(i_trip,7);
        
        s_p = loc_to_state( param, a.pickup_x, a.pickup_y);
        s_d = loc_to_state( param, a.dropoff_x, a.dropoff_y);
        idx_a = (s_p-1)*param.Ns + s_d;

        for i_start = 1:param.Ns
            
            tts = calcTTS(param, i_start, a);
            r = rev - param.C1*(tts + ttc); 
            
            R( i_start, s_d, idx_a) = R( i_start, s_d, idx_a) + ...
                r*param.lambda^(tf-tp);
            normalizing_denom( i_start, s_d, idx_a) = ...
                normalizing_denom( i_start, s_d, idx_a) + param.lambda^(tf-tp); 
        end
    end
    normalizing_denom(normalizing_denom == 0 ) = 1;
    R = R./normalizing_denom;
end