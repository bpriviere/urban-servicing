function writeResults(param, sim_results)

    if param.algo == 0
        write_filename_q = param.c_output_q_filename;
        write_filename_r = param.c_output_r_filename;
        write_filename_a = param.c_output_a_filename;
    elseif param.algo == 1
        write_filename_q = param.dist_ms_output_q_filename;
        write_filename_r = param.dist_ms_output_r_filename;
        write_filename_a = param.dist_ms_output_a_filename;
    elseif param.algo == 2
        write_filename_q = param.dist_ss_output_q_filename;
        write_filename_r = param.dist_ss_output_r_filename;
        write_filename_a = param.dist_ss_output_a_filename;
    end
    
    % save Q
    Q = sim_results.Q;
    [Nq, Ni, Nt] = size(Q);
    if param.algo == 0
        dlmwrite(write_filename_q, reshape(squeeze(Q(:,1,:)),[Nq, Nt]));
    else
        qstack = zeros(Nq*Ni,Nt);
        for i = 1:Ni
            start_idx = (i-1)*Nq + 1;
            end_idx = start_idx + Nq - 1;
            qstack(start_idx:end_idx,:) = squeeze(Q(:,i,:));
        end
        dlmwrite(write_filename_q, qstack);
    end
    
    % save rewards
    dlmwrite(write_filename_r, sim_results.R)
    
    % save agent positions
    agent_pos = zeros( 2*Ni, Nt);
    A = sim_results.A; % 2, param.Ni, param.Nt
    for i = 1:Ni
        idx = 2*(i-1) + 1;
        agent_pos( idx,: ) = A(1, i, :);
        agent_pos( idx + 1,: ) = A(2, i, :);
    end
    dlmwrite(write_filename_a, agent_pos);
    
end