

function kappa = get_kappa( param)

    eps = 0;
    delta = max( ...
        abs(param.max_R_online - param.min_R_MDP),...
        abs(param.min_R_online - param.max_R_MDP));

    d = eps*param.gamma*param.max_R_MDP / (1 - param.gamma^2) + ...
        delta / (1 - param.gamma);
    
    kappa = 4*d/(1-param.gamma);

end