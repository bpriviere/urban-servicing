function [Q_MDP, P_MDP, R_MDP, param] = solve_MDP(param, data)

    P_MDP = get_MDP_P(param, data);
    R_MDP = get_MDP_R(param, data);

    [V, policy, iter, cpu_time] = ...
        mdp_policy_iteration_modified(P_MDP, R_MDP, param.gamma);

    Q_MDP = get_MDP_Q(param, V, R_MDP);

    Q_MDP_T = Q_MDP';
    Q_MDP = Q_MDP_T(:);

    param.max_R_MDP = max(max(max(R_MDP)));
    param.min_R_MDP = min(min(min(R_MDP)));
    param.max_R_online = param.max_R_MDP;
    param.min_R_online = param.min_R_MDP;
    param.R_MDP = R_MDP;
    
end

