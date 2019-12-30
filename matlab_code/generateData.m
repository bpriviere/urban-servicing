
function data = generateData(param, filename)

    data = zeros( param.Nt*param.Nr, 6);
    
    phase_shift = randperm( param.Ns*param.Ns)/param.Ns^2;
    
    reward_a = @(a,t) param.R0 + param.A0*sin(...
        2*pi*phase_shift(a) + 2*pi*2*t/param.Nt);
    
    for i = 1:(param.Nt*param.Nr)
        t = i/param.Nr;
        
        s1 = randi(param.Ns); %Indexed at 1
        s2 = randi(param.Ns);
        a = (s1-1)*param.Ns + s2;
        
        data(i,5) = reward_a(a,t);
        data(i,6) = 1; % ttc task can just be constant for now
        data(i,7) = t;
        
%         if and(s2 ==1, strcmp(filename, param.input_data_filename))
%             data(i,5) = -param.R0/2 + t/param.Nt * param.R0* 4;
%         end
        
        % now assign a random location inside those cells:
        [x1,y1] = s_to_loc(param,s1);
        [x2,y2] = s_to_loc(param,s2);        
        data(i,1) = x1 + rand - 1/2;
        data(i,2) = y1 + rand - 1/2;
        data(i,3) = x2 + rand - 1/2;
        data(i,4) = y2 + rand - 1/2;
        
        if or(data(i,1) > param.Nx , or(data(i,3) > param.Nx, or(data(i,1) < 0 , data(i,3) < 0)))
            'help';
        end
        
        if or(data(i,2) > param.Ny , or(data(i,4) > param.Ny, or(data(i,2) < 0 , data(i,4) < 0)))
            'help';
        end
        
    end
    
%     writematrix(data, filename)
%     dlmwrite(filename,data);
%     csvread(filename);
    % write: [pickup, dropoff, revenue, ttc]
end

