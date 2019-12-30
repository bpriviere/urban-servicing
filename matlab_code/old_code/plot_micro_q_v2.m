function plot_micro_q_v2(param, Q_MPI, Q_no_MPI, Q_c)

    if param.Nq <= 8

        f = figure();
        t = 1:param.Nt;
            
        for i = 1:param.Ni
            for q_idx = 1:param.Nq
                subplot(param.Ns*param.Ns, param.Ns, q_idx);
                [s0,s1,s2] = q_to_sa(param,q_idx);
                xlabel('t'); ylabel(sprintf('Q(%d,(%d,%d))',s0,s1,s2))
                grid on
                plot(t,squeeze(Q_MPI(q_idx,i,:)), ...
                    'color', param.trace_color_h, 'DisplayName', sprintf('MPI: %d', i));
            end
        end
        
        for q_idx = 1:param.Nq
            subplot(param.Ns*param.Ns, param.Ns, q_idx);
%             [s0,s1,s2] = q_to_sa(param,q_idx);
%             xlabel('t'); ylabel(sprintf('Q_c(%d,(%d,%d))',s0,s1,s2))
            grid on
            plot(t,squeeze(Q_c(q_idx,1,:)), 'color', param.trace_color_c, ...
                'DisplayName', 'C','linewidth',5);
        end
         
        for i = 1:param.Ni
            for q_idx = 1:param.Nq
                ax(q_idx) = subplot(param.Ns*param.Ns, param.Ns, q_idx);
                plot(t,squeeze(Q_no_MPI(q_idx,i,:)), 'color', param.trace_color_d, 'DisplayName', sprintf('NO_MPI: %d', i));
                grid on
            end
        end
        linkaxes(ax,'y');

        h1 = plot(NaN,NaN,'color', param.trace_color_c);
        h2 = plot(NaN,NaN,'color', param.trace_color_h);
        h3 = plot(NaN,NaN,'color', param.trace_color_d);
        legend([h1,h2,h3],'Central','With Update','No Update','location','best')
        
        subplot(param.Ns*param.Ns, param.Ns, 1);
        title('Q Tracking')
        
        for i = 1:param.Nq
            subplot(param.Ns*param.Ns, param.Ns, i)
            set(gca,'FontSize',15);
        end
    %     set(f, 'Position', param.figure_pos);
    end

end