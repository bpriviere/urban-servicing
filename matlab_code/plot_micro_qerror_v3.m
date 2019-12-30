function plot_micro_qerror_v3(param, err_q_c, err_q_d_wupdate, err_q_d_woupdate)

    if param.Nq <= 8

        f = figure();
        t = 1:param.Nt;
            
        for i = 1:param.Ni
            for q_idx = 1:param.Nq
                ax(q_idx) = subplot(param.Ns*param.Ns, param.Ns, q_idx);
                [s0,s1,s2] = q_to_sa(param,q_idx);
                xlabel('t'); ylabel(sprintf('Q(%d,(%d,%d))',s0,s1,s2))
                grid on
                plot(t,squeeze(err_q_d_woupdate(q_idx,i,:)), ...
                    'color', param.trace_color_d_wou, 'linewidth',param.linewidth);
            end
        end
        
        for i = 1:param.Ni
            for q_idx = 1:param.Nq
                subplot(param.Ns*param.Ns, param.Ns, q_idx);
                [s0,s1,s2] = q_to_sa(param,q_idx);
                grid on
                plot(t,squeeze(err_q_d_wupdate(q_idx,i,:)), ...
                    'color', param.trace_color_d_wu, 'linewidth', param.linewidth) 
            end
        end        
        
        for q_idx = 1:param.Nq
            subplot(param.Ns*param.Ns, param.Ns, q_idx);
%             [s0,s1,s2] = q_to_sa(param,q_idx);
%             xlabel('t'); ylabel(sprintf('Q_c(%d,(%d,%d))',s0,s1,s2))
            grid on
            plot(t,squeeze(err_q_c(q_idx,1,:)), 'color', param.trace_color_c_wou, ...
                'linewidth',param.linewidth);
        end
        
        linkaxes(ax,'y');

        h1 = plot(NaN,NaN,'color', param.trace_color_c_wou,'linewidth',param.linewidth);
        h2 = plot(NaN,NaN,'color', param.trace_color_d_wu,'linewidth',param.linewidth);
        h3 = plot(NaN,NaN,'color', param.trace_color_d_wou,'linewidth',param.linewidth);
        legend([h1,h2,h3],'Central','D: w Update','D: w/o Update','location','best')
        
        subplot(param.Ns*param.Ns, param.Ns, 1);
        title('Q Tracking Error')
        
        for i = 1:param.Nq
            subplot(param.Ns*param.Ns, param.Ns, i)
            set(gca,'FontSize',param.fontsize);
        end
        
    end

end