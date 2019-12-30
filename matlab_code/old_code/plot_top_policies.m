function plot_top_policies(param, Q_MPI, Q_no_MPI, Q_c)


figure()
xlabel('t'), 

for i_t = 1:param.Nt
    
    for i = 1:1 %param.Ni
        
        [~,idx_max_mpi] = max(Q_MPI(:,i,i_t));
        [~,idx_max_c] = max(Q_c(:,i,i_t));
        [~,idx_max_no_mpi] = max(Q_no_MPI(:,i,i_t));
        
        plot([i_t-1, i_t], idx_max_mpi*[1,1], 'b','linewidth',5);
        plot([i_t-1, i_t], idx_max_no_mpi*[1,1], 'g','linewidth',5);
        plot([i_t-1, i_t], idx_max_c*[1,1], 'k','linewidth',5);
        
    end
end

grid on

% for i_q = 1:param.Nq
%     
% end

end
