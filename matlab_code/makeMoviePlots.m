function makeMoviePlots(param, agents, requests, k)

    fn_base = sprintf('%s%s_', param.movie_filename, num2str(k));
    
    nsteps = 20;
    steps = [1:nsteps]/nsteps;
    % make lines from current location to pickup point
    line_to_pickup = zeros(param.Ni, 2, nsteps);
    line_to_dropoff = zeros(param.Ni, 2, nsteps);
    for i = 1:param.Ni
        line_to_pickup(i,1,:) = (1-steps)*agents(i).x + ...
            steps*requests(agents(i).curr_action_i).pickup_x;
        line_to_pickup(i,2,:) = (1-steps)*agents(i).y + ...
            steps*requests(agents(i).curr_action_i).pickup_y;
        line_to_dropoff(i,1,:) = (1-steps)*requests(agents(i).curr_action_i).pickup_x + ...
            steps*requests(agents(i).curr_action_i).dropoff_x;
        line_to_dropoff(i,2,:) = (1-steps)*requests(agents(i).curr_action_i).pickup_y + ...
            steps*requests(agents(i).curr_action_i).dropoff_y;
    end
    
    f = figure();
    set(f,'Visible','off');
    for i_step = 1:nsteps
        clf
        
        if param.real_data_on
            mapshow(param.map, 'FaceColor', param.face_color,'EdgeColor',[0.75 0.75 0.75]) 
            axis square
            axis off
        else
            axis equal
            axis(param.scale*[ 0, param.Nx 0 param.Ny])
            set(gca,'XTick', [0:1:param.Nx]);
            set(gca,'YTick', [0:1:param.Ny]);
            grid on
        end
        
        for i = 1:param.Ni
            h = plot( line_to_pickup(i,1,i_step), line_to_pickup(i,2,i_step), param.agent_marker);
            set(h, 'MarkerFaceColor', get(h,'Color')); 
            if i == 1
                plot( line_to_pickup(i,1,i_step) + param.R_task/param.scale*cos([0:0.01:2*pi]), ...
                    line_to_pickup(i,2,i_step) + param.R_task/param.scale*sin([0:0.01:2*pi]), 'b--');
            end
        end

        for i = 1:param.Ni
            j = agents(i).curr_action_i;
            if requests(j).revenue ~= 0
                plot( requests(j).pickup_x, requests(j).pickup_y, param.request_marker)
            end
        end
        
        h1 = plot(NaN, NaN, param.agent_marker);
        h2 = plot(NaN, NaN, param.request_marker);
        legend([h1,h2],'Agent','Customer','location','SouthWest')
        
        fn_complete = sprintf('%s%s.jpg', fn_base, num2str(i_step));
        saveas(f, fn_complete);
    end

    for i_step = 1:nsteps
        clf
        
        if param.real_data_on
            mapshow(param.map, 'FaceColor', param.face_color,'EdgeColor',[0.75 0.75 0.75]) 
            axis square
            axis off
        else
            axis equal
            axis(param.scale*[ 0, param.Nx 0 param.Ny])
            set(gca,'XTick', [0:1:param.Nx]);
            set(gca,'YTick', [0:1:param.Ny]);
            grid on
        end        
        
        for i = 1:param.Ni
            h = plot( line_to_dropoff(i,1,i_step), line_to_dropoff(i,2,i_step), param.agent_marker);
            set(h, 'MarkerFaceColor', get(h,'Color')); 
            if i == 1
                plot( line_to_dropoff(i,1,i_step) + param.R_task/param.scale*cos([0:0.01:2*pi]), ...
                    line_to_dropoff(i,2,i_step) + param.R_task/param.scale*sin([0:0.01:2*pi]), 'b--');
            end
        end

        for i = 1:param.Ni
            j = agents(i).curr_action_i;
            if requests(j).revenue ~= 0
                plot( requests(j).dropoff_x, requests(j).dropoff_y, param.request_marker_dropoff)
            end
        end

        h1 = plot(NaN, NaN, param.agent_marker);
        h2 = plot(NaN, NaN, param.request_marker_dropoff);
        legend([h1,h2],'Agent','Customer','location','SouthWest')
        
        fn_complete = sprintf('%s%s.jpg', fn_base, num2str(i_step + nsteps));
        saveas(f, fn_complete);
    end
    

end