function plot_action_freq(param, ACT)

freq = NaN(param.Nq, param.Nt);

for i_q = 1:param.Nq
    for i_t = 1:param.Nt
        sum = 0 ;
        for i_i = 1:param.Ni
            if ACT(i_i, i_t) == i_q
                sum = sum + 1;
            end
        end
        freq(i_q, i_t) = sum;
    end
end

t = 1:param.Nt;
figure()
for i_q = 1:param.Nq
    [s0,s1,s2] = q_to_sa(param, i_q);
    plot(t, freq(i_q,:),'.','MarkerSize',20,...
        'DisplayName',sprintf('Q(%d,(%d,%d))', s0, s1, s2))
end
legend('location','best')
end