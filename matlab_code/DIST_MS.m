function agents = DIST_MS(param,agents)

    A = getA(param, agents);
%     F = getF(param);
%     invF = inv(F);
    Q = getQ(param);
    invQ = inv(Q);
    R = getR(param);
    invR = inv(R);
    
    Q_kp1 = zeros(param.Nq, param.Ni);
    P_kp1 = zeros(param.Nq, param.Nq, param.Ni);
    I_kp1 = zeros(param.Nq, param.Nq, param.Ni);
    i_kp1 = zeros(param.Nq, param.Ni);
    U_kp1 = zeros(param.Nq, param.Nq, param.Ni);
    u_kp1 = zeros(param.Nq, param.Ni);

    for i = 1:param.Ni
        
        x_km1km1 = agents(i).Q;
        P_km1km1 = agents(i).P;
        
        F = getF2(param, agents(i).z);
%         F = getF(param);
        invF = inv(F);
        
        % information states
        Y_km1km1 = inv(P_km1km1);
        y_km1km1 = Y_km1km1*x_km1km1;
        
        % predict
%         M = invF'*Y_km1km1*invF;
%         S = eye(param.Nq) - M*inv(M + invQ);
%         Z_kkm1 = S*M;
%         z_kkm1 = S*invF'*y_km1km1;
        M = invF'*Y_km1km1*invF;
        C = M*inv(M + invQ);
        L = eye(param.Nq) - C;
        Y_kkm1 = L*M*L' + C*invQ*C';
        y_kkm1 = L*inv(F)'*y_km1km1;
        
        % get info
%         H = squeeze(getH(param, agents(i).z));
        H = squeeze(get_H2(param,agents(i).z));
        
        % innovate
        info_vec = H'*invR* agents(i).z;
        info_mat = H'*invR*H;
        % consensus
        u_vec = get_update_u(param, agents, info_vec, A, i);
        U_mat = get_update_U(param, agents, info_mat, A, i);
        y_kk = y_kkm1 + param.Ni*u_vec;
        Y_kk = Y_kkm1 + param.Ni*U_mat;
        P_k = inv(Y_kk);
        x_k = P_k*y_kk;
        
        Q_kp1(:,i) = x_k;
        P_kp1(:,:,i) = P_k;
        i_kp1(:,i) = info_vec;
        I_kp1(:,:,i) = info_mat;
        u_kp1(:,i) = u_vec;
        U_kp1(:,:,i) = U_mat;
    end
    
    % update agents
    for i = 1:param.Ni
        agents(i).Q = Q_kp1(:,i);
        agents(i).P = squeeze(P_kp1(:,:,i));
        agents(i).info_vec = i_kp1(:,i);
        agents(i).info_mat = squeeze(I_kp1(:,:,i));        
        agents(i).u = u_kp1(:,i);
        agents(i).U = squeeze(U_kp1(:,:,i));
    end
end