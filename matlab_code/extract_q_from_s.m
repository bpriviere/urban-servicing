function q_extracted = extract_q_from_s(param,s,q)

    start_idx = (s-1)*param.Ns*param.Ns+1;
    end_idx = start_idx+param.Ns*param.Ns-1;
    q_extracted = q(start_idx:end_idx);

end