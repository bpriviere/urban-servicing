function plot_macro_q(Jstar, R_CASE, N_CASE)

    figure()
    for i = 1:length(R_CASE)
        plot( N_CASE, Jstar(i,:), 's-', 'DisplayName',sprintf('R_{comm}: %s', num2str(R_CASE(i))));
    end
    legend('location','best');
    xlabel('Number of Agents');
    ylabel('Normalized J*');   
    grid on
%     axis([0 N_CASE(end) 0 1]);

end