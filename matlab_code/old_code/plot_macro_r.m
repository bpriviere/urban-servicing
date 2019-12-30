function plot_macro_r(param)

    f = figure();
    t = 1:param.Nt*param.n_episodes;
    if param.cent_on
        R_c = dlmread(param.c_output_r_filename);
%         subplot(1,2,1)
%         plot(t,R_c, 'DisplayName', sprintf('CENT'));
%         subplot(1,2,2)
        plot(t,cumsum(R_c), 'DisplayName', sprintf('CENT'));
    end
    
    if param.dist_ss_on
        R_d_ss = dlmread(param.dist_ss_output_r_filename); 
%         subplot(1,2,1)
%         plot(t,R_d_ss, 'DisplayName', sprintf('DIST SS'));
%         subplot(1,2,2)
        plot(t,cumsum(R_d_ss), 'DisplayName', sprintf('DIST SS'));
    end
    
    if param.dist_ms_on
        R_d_ms = dlmread(param.dist_ms_output_r_filename);
%         subplot(1,2,1)
%         plot(t,R_d_ms, 'DisplayName', sprintf('DIST MS'));
%         subplot(1,2,2)
        plot(t,cumsum(R_d_ms), 'DisplayName', sprintf('DIST MS'));
    end
    
    if param.greedy_on
        R_greedy = dlmread(param.greedy_output_r_filename);
%         subplot(1,2,1)
%         plot(t,R_greedy, 'DisplayName', sprintf('GREEDY'));
%         subplot(1,2,2)
        plot(t,cumsum(R_greedy), 'DisplayName', sprintf('GREEDY'));
    end
    
    legend('location','best')
%     subplot(1,2,1)
%     ylabel('Profit [USD]')
%     xlabel('Timestep')
%     grid on
%     subplot(1,2,2)
    ylabel('Cumulative Profit [USD]')
    xlabel('Timestep')
    grid on


end