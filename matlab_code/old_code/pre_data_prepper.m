
function param = pre_data_prepper(param)

    sec2min = 60;
    min2hour = 60;
    hour2day = 24;

    % online data
    data = csvread(param.raw_online_filename);

    data_sorted = sortrows(data,[7,8,9,10]);
    data_sorted(:,7) = data_sorted(:,7) + data_sorted(:,8)/hour2day + ...
        data_sorted(:,9)/(min2hour*hour2day) + data_sorted(:,10)/(sec2min*min2hour*hour2day);
    
    data_sorted = data_sorted(:,1:7);
    idx = ~isnan(data_sorted(:,1));
    data_sorted = data_sorted(idx,:); % remove nans
    hold = [data_sorted(:,1), data_sorted(:,3)]; % switch columsn
    data_sorted(:,1) = data_sorted(:,2);
    data_sorted(:,3) = data_sorted(:,4);
    data_sorted(:,2) = hold(:,1);
    data_sorted(:,4) = hold(:,2);

    count_freq = zeros(param.Ns_map,1);
    for i = 1:size(data_sorted,1)
        loc = data_sorted(i,:);
        pickup_state = loc_to_state_pp(param, loc(1), loc(2));
        dropoff_state = loc_to_state_pp(param, loc(3), loc(4));
        count_freq(pickup_state) = count_freq(pickup_state) + 1;
        count_freq(dropoff_state) = count_freq(dropoff_state) + 1;
    end

    % remove low number of trips
    [~, descending_idx] = sort(count_freq,'descend');
    idx_des = descending_idx(1:param.n_state_thresh);
    good_s_idx = zeros(size(count_freq,1),1);
    good_s_idx(idx_des) = 1; 
    data_trimmed = NaN(size(data_sorted));
    count = 1;
    for i = 1:size(data_sorted,1)
        s1 = loc_to_state_pp(param, data_sorted(i,1), data_sorted(i,2));
        s2 = loc_to_state_pp(param, data_sorted(i,3), data_sorted(i,4));
        if and(good_s_idx(s1), good_s_idx(s2))
            data_trimmed(count,:) = data_sorted(i,:);
            count = count + 1;
        end
    end
    idx = ~isnan(data_trimmed(:,1));
    data_trimmed = data_trimmed(idx,:);
    param.good_s_idx = good_s_idx;
    param.data = data_trimmed;
    dlmwrite(param.input_online_filename, data_trimmed);
    
        % training
    data = csvread(param.raw_training_filename);

    data_sorted = sortrows(data,[7,8,9,10]);
    data_sorted(:,7) = data_sorted(:,7) + data_sorted(:,8)/hour2day + ...
        data_sorted(:,9)/(min2hour*hour2day) + data_sorted(:,10)/(sec2min*min2hour*hour2day);
    data_sorted = data_sorted(:,1:7);    
    idx = ~isnan(data_sorted(:,1));
    data_sorted = data_sorted(idx,:); % remove nans
    hold = [data_sorted(:,1), data_sorted(:,3)]; % switch columsn
    data_sorted(:,1) = data_sorted(:,2);
    data_sorted(:,3) = data_sorted(:,4);
    data_sorted(:,2) = hold(:,1);
    data_sorted(:,4) = hold(:,2);
    
    data_trimmed = NaN(size(data_sorted));
    count = 1;
    for i = 1:size(data_sorted,1)
        s1 = loc_to_state_pp(param, data_sorted(i,1), data_sorted(i,2));
        s2 = loc_to_state_pp(param, data_sorted(i,3), data_sorted(i,4));
        if and(good_s_idx(s1), good_s_idx(s2))
            data_trimmed(count,:) = data_sorted(i,:);
            count = count + 1;
        end
    end
    idx = ~isnan(data_trimmed(:,1));
    data_trimmed = data_trimmed(idx,:);    
    param.training_data = data_trimmed;
%     dlmwrite(param.input_training_filename, data_trimmed);
   
end

    