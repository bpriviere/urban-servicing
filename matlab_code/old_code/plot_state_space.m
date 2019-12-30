function plot_state_space(param, agents, k)


    f = figure();

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
        h = plot( agents(i).x, agents(i).y, param.agent_marker);
        set(h,'MarkerFaceColor',get(h,'Color'))
    end
    plot( agents(1).x + param.R_task/param.scale*cos([0:0.01:2*pi]), ...
              agents(1).y + param.R_task/param.scale*sin([0:0.01:2*pi]), 'b--');
    
    
    start_idx = (k-1)*param.Nr+1;
    end_idx = start_idx + param.Nr-1;
    data = param.data(start_idx:end_idx,:);
    
    for j = 1:param.Nr
        plot( data(j,1), data(j,2), param.request_marker)
    end
    
    h1 = plot(NaN, NaN, param.agent_marker);
    h2 = plot(NaN, NaN, param.request_marker);
    legend([h1,h2],'Agent','Customer','location','SouthWest')
    
end

