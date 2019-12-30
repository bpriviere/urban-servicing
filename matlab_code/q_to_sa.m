function [s0,s1,s2] = q_to_sa(param,q_idx)
    q_idx = q_idx - 1;
    s0 = floor( q_idx/param.Ns^2) + 1;
    rem = mod(q_idx,param.Ns^2);
    s1 = floor( rem/param.Ns) + 1;
    s2 = mod( rem, param.Ns) + 1; 
end