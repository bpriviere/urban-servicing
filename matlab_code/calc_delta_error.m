function delta_err = calc_delta_error(param, k)
    
%     kappa = get_kappa( param);    
%     delta_q = param.delta_z_max / min(param.delta_f, param.underbar_alpha);
%     delta_err = kappa + (k - param.last_update_k)*(param.delta_z_max + delta_q);

    delta_s = param.Ni * param.delta_env / param.underbar_lambda2L*...
        (1 - (1 - param.underbar_lambda2L)^(k - param.last_update_k));
    delta_h = 2*sqrt(param.Nq*param.ProcessNoise)/(param.underbar_alpha*(1-param.gamma));
    delta_err = delta_s + delta_h;
    
%     fprintf('delta_s: %d\n', delta_s)
%     fprintf('delta_h: %d\n', delta_h)
%     fprintf('delta_err: %d\n', delta_err)
end