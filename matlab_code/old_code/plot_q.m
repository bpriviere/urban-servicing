function plot_q_c(param, sim_results)

    figure()
    Q = sim_results.Q; % Nq, Ni, Nt
    t = 1:param.Nt;
    for q_idx = 1:param.Nq
        subplot(param.Ns*param.Ns, param.Ns, q_idx);
        [s0,s1,s2] = q_to_sa(param,q_idx);
        xlabel('t'); ylabel(sprintf('Q(%d,(%d,%d))',s0,s1,s2))
        grid on
        plot(t,squeeze(Q(q_idx,i_idx,:)), 'DisplayName', 'Central');
        legend('location','best')
    end

    
end