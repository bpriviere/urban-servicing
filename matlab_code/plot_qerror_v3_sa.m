function plot_qerror_v3_sa(param, err_q_c, err_q_d_wupdate, err_q_d_woupdate, s0,s1,s2)

    figure()
    t = 1:param.Nt;
    q_idx = param.Ns^2*(s0-1) + param.Ns*(s1-1) + s2;
    for i = 1:param.Ni
        xlabel('t'); ylabel(sprintf('Q(%d,(%d,%d))',s0,s1,s2))
        grid on
        plot(t,squeeze(err_q_d_woupdate(q_idx,i,:)), ...
            'color', param.trace_color_d_wou, 'linewidth',param.linewidth);
    end

    for i = 1:param.Ni
        grid on
        plot(t,squeeze(err_q_d_wupdate(q_idx,i,:)), ...
            'color', param.trace_color_d_wu, 'linewidth', param.linewidth) 
    end        

    grid on
    plot(t,squeeze(err_q_c(q_idx,1,:)), 'color', param.trace_color_c_wou, ...
        'linewidth',param.linewidth);
    
    yl = ylim;
    K = 0:param.k:t(end);
    for i_k = 1:length(K)
        plot( [K(i_k), K(i_k)], [yl(1), yl(2)],'r--','linewidth',param.linewidth)
    end

    h1 = plot(NaN,NaN,'color', param.trace_color_c_wou,'linewidth',param.linewidth);
    h2 = plot(NaN,NaN,'color', param.trace_color_d_wu,'linewidth',param.linewidth);
    h3 = plot(NaN,NaN,'color', param.trace_color_d_wou,'linewidth',param.linewidth);
    h4 = plot(NaN,NaN,'r--','linewidth',param.linewidth);
    legend([h1,h2,h3,h4],'Central','D: w Update','D: w/o Update','Update','location','best')
    
    title('Q Tracking Error')
    set(gca,'FontSize',param.fontsize);
end