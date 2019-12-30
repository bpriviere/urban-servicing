function requests = getRequests(param,k)

    requests = {};
    
    R1 = (k-1)*param.Nr+1;
    R2 = R1 + param.Nr-1;
    data = param.data(R1:R2,:);
    
%     RANGE = [R1 0 R2 5];    
%     data = dlmread(param.data_filename, ',', RANGE);
%     data = data([R1:R2],:);
    
    for j = 1:param.Nr
        requests(j).idx = j;
        requests(j).pickup_x = data(j,1);
        requests(j).pickup_y = data(j,2);
        requests(j).dropoff_x = data(j,3);
        requests(j).dropoff_y = data(j,4);     
        requests(j).revenue = data(j,5);
        requests(j).ttc = data(j,6);
        requests(j).t = data(j,7);
    end
end