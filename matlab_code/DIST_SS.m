function agents = DIST_SS(param,agents)

    A = getA(param, agents);
    Q = getQ(param); % covariance process
    R = getR(param); % covariance measurement
    
    Q_kp1 = zeros(param.Nq, param.Ni);
    P_kp1 = zeros(param.Nq, param.Nq, param.Ni);
    for i = 1:param.Ni
        
        if param.fix_on 
            F = getF2(param, agents(i).z);
            H_i = squeeze(get_H2(param, agents(i).z));
        else
            F = getF(param);
            H_i = squeeze(getH(param, agents(i).z));
        end
            
        % predict 
        P_kkm1 = F*agents(i).P*F' + Q;
        
        % innovate
        S = R + H_i*P_kkm1*H_i';
        invS = inv(S);
        
        % learning rate/kalman gain 
        Alpha = P_kkm1*H_i'*invS;
        
        % enforce base Alpha
        for i_q = 1:param.Nq
            if Alpha(i_q,i_q) < param.force_alpha
                Alpha(i_q,i_q) = param.force_alpha;
            end
        end
        
        % consensus term 
        consensus_term = zeros(param.Nq,1);
        for j = 1:agents(i).nn
            neighbor = agents(j);
            consensus_term = consensus_term + ...
                A(i,neighbor.i)*(neighbor.Q - agents(i).Q);
        end
        
        % update law
        Q_kp1(:,i) = (F - Alpha)*agents(i).Q + ...
            Alpha*agents(i).z + consensus_term;
        
        P_kp1(:,:,i) = (eye(param.Nq) - Alpha*H_i)*P_kkm1;

    end
    
    % update agents
    for i = 1:param.Ni
        agents(i).Q = Q_kp1(:,i);
        agents(i).P = squeeze(P_kp1(:,:,i));
        agents(i).alpha = Alpha(Alpha ~= 0);
    end
end