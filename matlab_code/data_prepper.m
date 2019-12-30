
% data prepper: 
% INPUT (RAW DATA): [Pickup centroid, dropoff centroid, trip_seconds, revenue, day, hour month, second]
% OUTPUT (CLEAN DATA): [Pickup centroid, dropoff centroid, trip_seconds, revenue] (chronological)

function param = data_prepper(param)

    data = NaN( 1500, 11);

    fid = fopen(param.raw_data_filename);
    count = 1;
    tline = fgetl(fid);

    while ischar(tline)
        coord_start_idx = find(tline == '[');
        coord_end_idx = find(tline == ']');
        data(count,1:2) = str2num(tline(coord_start_idx(1):coord_end_idx(1)));
        data(count,3:4) = str2num(tline(coord_start_idx(2):coord_end_idx(2)));
        data(count,5:end) = str2num(tline((coord_end_idx(2)+4):end));

    %     rest = str2num(tline((coord_end_idx(2)+4):end));
    %     data(i,5:6) = rest(1:2);
        count = count + 1;
        tline = fgetl(fid);
    end

    data_sorted = sortrows(data, [7,8,9,10,11]);
    data_sorted = data_sorted(:,1:6);
    idx = ~isnan(data_sorted(:,1));
    data_sorted = data_sorted(idx,:);
    

    % count frequency of trips:
%     my_set_param;
    count_freq = zeros(param.Ns_map,1);
    for i = 1:size(data_sorted,1)
        loc = data_sorted(i,:);
        pickup_state = loc_to_state_pp(param, loc(1), loc(2));
        dropoff_state = loc_to_state_pp(param, loc(3), loc(4));
        count_freq(pickup_state) = count_freq(pickup_state) + 1;
        count_freq(dropoff_state) = count_freq(dropoff_state) + 1;
    end

    % remove low number of trips
    good_s_idx = count_freq > param.state_thresh;
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
    dlmwrite(param.input_data_filename, data_trimmed);

end
