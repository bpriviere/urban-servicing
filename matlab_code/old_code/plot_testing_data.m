

% this is to test input data:

function plot_testing_data(param, data)


    figure()
    
    for i_trip = 1:size(data,1)
        s1 = loc_to_state(param, data(i_trip,1), data(i_trip,2));
        s2 = loc_to_state(param, data(i_trip,3), data(i_trip,4));
        
        a.pickup_x = data(i_trip,1);
        a.pickup_y = data(i_trip,2);
        a.dropoff_x = data(i_trip,3);
        a.dropoff_y = data(i_trip,4);
        idx_a = (s1-1)*param.Ns + s2;

        rev = data(i_trip,5);
        ttc = data(i_trip,6);
        tp = data(i_trip,7);      
        
        for i_state = 1:param.Ns
            tts = calcTTS(param, i_state, a);
            r = rev - param.C1*(tts + ttc); 
            
            q_idx = (i_state-1)*param.Ns*param.Ns + (s1-1)*param.Ns + s2;
            
            subplot(param.Ns*param.Ns, param.Ns, q_idx)
            plot( tp, r, 'b.')
        end
        
        
    end



end