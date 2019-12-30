function plot_micro_q(param)

    if param.Nq <= 8

        f = figure();
        t = 1:param.Nt;
        if param.cent_on
            Q_c = dlmread(param.c_output_q_filename);
            
            for q_idx = 1:param.Nq
                ax(q_idx) = subplot(param.Ns*param.Ns, param.Ns, q_idx);
                [s0,s1,s2] = q_to_sa(param,q_idx);
                xlabel('t'); ylabel(sprintf('Q(%d,(%d,%d))',s0,s1,s2))
                grid on
                plot(t,squeeze(Q_c(q_idx,:)), 'DisplayName', 'CENT');
            end
            linkaxes(ax,'y');
        end

        if param.dist_ss_on
            Q_d_ss = dlmread(param.dist_ss_output_q_filename);            
            for i = 1:param.Ni
                for q_idx = 1:param.Nq
                    ax(q_idx) = subplot(param.Ns*param.Ns, param.Ns, q_idx);
                    idx = (i-1)*param.Nq + q_idx;
                    plot(t,Q_d_ss(idx,:), 'DisplayName', sprintf('DIST SS: %d', i));
                    grid on
                end
            end
            linkaxes(ax,'y');
        end

        if param.dist_ms_on
            Q_d_ms = dlmread(param.dist_ms_output_q_filename);
            for i = 1:param.Ni
                for q_idx = 1:param.Nq
                    subplot(param.Ns*param.Ns, param.Ns, q_idx);
                    idx = (i-1)*param.Nq + q_idx;
                    plot(t,Q_d_ms(idx,:), 'DisplayName', sprintf('DIST MS: %d', i));
                    grid on
                end
            end
        end

        if param.Ni < 3
            legend('location','best')
        end
        
        subplot(param.Ns*param.Ns, param.Ns, 1);
        title('Q Tracking')
    %     set(f, 'Position', param.figure_pos);
    end

end