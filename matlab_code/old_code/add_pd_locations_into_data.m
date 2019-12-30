
% treat raw data

function data_w_pd = add_pd_locations_into_data(param, data)

    data_w_pd = zeros(size(data,1), 7);
    for i = 1:size(data,1)
        [x_p,y_p] = s_to_loc_pp(param,data(i,1));
        [x_d,y_d] = s_to_loc_pp(param,data(i,2));
        data_w_pd(i,:) = [x_p, y_p, x_d, y_d, data(i,3), data(i,4), i];
    end

end