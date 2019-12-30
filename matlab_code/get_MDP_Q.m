
function Q_MDP = get_MDP_Q(param, V, R)

    % make Q values
    Q_MDP = zeros(param.Ns, param.Ns*param.Ns);
    
    for i_start = 1:param.Ns % start
        for i_sp = 1:param.Ns % pickup
            for i_sd = 1:param.Ns % dropoff
                i_a = (i_sp-1)*param.Ns + i_sd;
                Q_MDP(i_start, i_a) = param.gamma*V(i_sd) + R(i_start, i_sd, i_a);
            end
        end
    end
end
