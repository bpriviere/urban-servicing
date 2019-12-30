function plot_state_space_v2(param, LOC, k)


    f = figure();
    hold on
    if param.real_data_on
        mapshow(param.map, 'FaceColor', param.face_color,'EdgeColor','k') 
        axis square
        axis off
    else
        axis equal
        axis(param.scale*[ 0, param.Nx 0 param.Ny])
        set(gca,'XTick', [0:1:param.Nx]);
        set(gca,'YTick', [0:1:param.Ny]);
        grid on
    end    
    
    start_idx = (k-1)*param.Nr+1;
    end_idx = start_idx + param.Nr-1;
    data = param.data(start_idx:end_idx,:);
    
    for j = 1:param.Nr
%         pl = plot( data(j,1), data(j,2), param.request_marker, ...
%             'MarkerSize',10,'MarkerFaceColor','y','FaceAlpha',0.2);
%         pl.Color(4) = 0.2;
        s1 = scatter( data(j,1), data(j,2), 100, param.request_marker, ...
            'MarkerFaceColor','y','MarkerFaceAlpha',0.4);
    end   
              
    
    for i = 1:param.Ni
        h = plot( LOC(1,i,k), LOC(2,i,k), param.agent_marker,'MarkerSize',param.markersize);
        set(h,'MarkerFaceColor',get(h,'Color'))
    end
    plot( LOC(1,1,k) + param.R_task/param.scale*cos([0:0.01:2*pi]), ...
              LOC(2,1,k) + param.R_task/param.scale*sin([0:0.01:2*pi]), 'b--',...
              'linewidth',param.linewidth);
    

%     h1 = plot(NaN, NaN, param.agent_marker,'MarkerFaceColor', 'b');
%     h2 = plot(NaN, NaN, param.request_marker,'color','y','MarkerFaceColor','y');
%     legend([h1,h2],'Agent','Customer','location','SouthEast')
    set(gca,'FontSize',20);
end

