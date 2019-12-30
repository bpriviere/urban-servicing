function plot_qbar(param)


    t = 1:param.Nt;
   
    f = figure();
    if param.dist_ss_on
        Q_d_ss = dlmread(param.dist_ss_output_q_filename);            
        Q_d_ss_reshape = my_reshape_2d_to_3d(param, Q_d_ss);
        Q_bar = squeeze(mean(Q_d_ss_reshape,2));
        
        for i = 1:param.Ni
            for q_idx = 1:param.Nq
                subplot(param.Ns*param.Ns, param.Ns, q_idx);
                q_tilde = abs(squeeze(Q_d_ss_reshape(q_idx,i,:)) ...
                    - Q_bar(q_idx,:)');
                plot(t, q_tilde, 'DisplayName', sprintf('DIST SS: %d', i));
                grid on
            end
        end
        subplot(param.Ns * param.Ns, param.Ns, 1)
        title('DIST SS: Deviation from Average Q')
    end
    
    f = figure();
    if param.dist_ss_on
        Q_d_ms = dlmread(param.dist_ms_output_q_filename);            
        Q_d_ms_reshape = my_reshape_2d_to_3d(param, Q_d_ms);
        Q_bar = squeeze(mean(Q_d_ms_reshape,2));
        
        for i = 1:param.Ni
            for q_idx = 1:param.Nq
                subplot(param.Ns*param.Ns, param.Ns, q_idx);
                q_tilde = abs(squeeze(Q_d_ms_reshape(q_idx,i,:)) ...
                    - Q_bar(q_idx,:)');
                plot(t, q_tilde, 'DisplayName', sprintf('DIST MS: %d', i));
                grid on
            end
        end
        subplot(param.Ns * param.Ns, param.Ns, 1)
        title('DIST MS: Deviation from Average Q')
    end    
     
     
         
%     f = figure();
%     if param.dist_ms_on
%         Q_d_ms = dlmread(param.dist_ms_output_q_filename);            
%         for i = 1:param.Ni
%             for q_idx = 1:param.Nq
%                 subplot(param.Ns*param.Ns, param.Ns, q_idx);
%                 idx = (i-1)*param.Nq + q_idx;
%                 plot(t,Q_d_ms(idx,:), 'DisplayName', sprintf('DIST SS: %d', i));
%                 grid on
%             end
%         end
%     end
        
    



end 