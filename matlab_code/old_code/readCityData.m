function [param, data] = readCityData(param, filename)
    
    % input into this 


    % read a file
    fprintf('Reading File...\n')
    nrows = param.Nr*param.Nt*20;
%     data = csvread(filename,1,1, [0 0 nrows 3]);
    data = csvread(filename);
%     if and( size(data,2) ~= 4, size(data,2) ~=7)
%         data = data(2:end,2:end);
%     end
    
    % sort by timestamp
    data = sort_by_timestamp( param, data);
    
    if size(data,2) ~= 7
        % add locations
        fprintf('Adding locations...\n')    
        data = add_pd_locations_into_data(param, data);
    end
    
    % reduce data
    fprintf('Reducing Data...\n')    
    [data, good_idx] = reduce_data(param, data);
    
    % reduce statespace
    fprintf('Reducing State Space...\n')    
    param = reduce_state_space( param, data, good_idx);
    
end

