
function param = preprocessing_state_reduction(param)

    % This function builds the reduced state to full map state vectors
    % by finding which states are not used frequently (data already
    % trimmed in data requests from data_prepper.m) 

%     param.Ns_map = size(param.map,1);
    
    count_freq = zeros(param.Ns_map, 1);
    for i = 1:size(param.data,1)
        loc = param.data(i,:);
        pickup_state = loc_to_state_pp(param, loc(1), loc(2));
        dropoff_state = loc_to_state_pp(param, loc(3), loc(4));
        count_freq(pickup_state) = count_freq(pickup_state) + 1;
        count_freq(dropoff_state) = count_freq(dropoff_state) + 1;
    end
    
%     good_s_idx = count_freq > param.state_thresh;
    good_s_idx = param.good_s_idx;
    param.Ns = sum(good_s_idx);
    param.my_state_to_map_state = zeros(param.Ns,1);
    param.map_state_to_my_state = zeros(param.Ns_map,1);
    count = 1;
    for i = 1:size(param.map,1)
        if good_s_idx(i)
            param.my_state_to_map_state(count) = i;
            param.map_state_to_my_state(i) = count;
            count = count + 1;
        end
    end
end