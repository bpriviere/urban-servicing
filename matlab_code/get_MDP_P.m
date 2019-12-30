function P = get_MDP_P(param, data)

    ntrips = size(data,1);
    P = zeros(param.Ns, param.Ns, param.Ns*param.Ns);
        
    for i = 1:ntrips
        s_pickup = loc_to_state( param, data(i,1), data(i,2));
        s_dropoff = loc_to_state( param, data(i,3), data(i,4));
        a = (s_pickup - 1)*param.Ns + s_dropoff;
        
        for i_s = 1:param.Ns
            P(i_s,s_dropoff,a) = 1;
        end
    end
    
%     for i_a = 1:(param.Ns*param.Ns)
%         for i_s = 1:param.Ns
%             P(i_s,:,i_a) = P(i_s,:,i_a)/sum(P(i_s,:,i_a),2);
%         end
%     end

end