function agents = CKIF(param, agents)
% centralized Kalman Information Filter

    % State, Covariance
    x_km1km1 = agents(1).Q;
    P_km1km1 = agents(1).P;

    % Information State
    Y_km1km1 = inv(P_km1km1);
    y_km1km1 = Y_km1km1*x_km1km1;

    % get measurements
    z_kj = zeros(param.Nq, param.Ni);
    for i = 1:param.Ni
        z_kj(:,i) = agents(i).z;
    end

    % get matrices
    if param.fix_on 
        F = getF2(param, z_kj);
        H = get_H2(param,z_kj);
    else
        F = getF(param);
        H = getH(param,z_kj);
    end
    invF = inv(F);
    Q = getQ(param);
    invQ = inv(Q);
    R = getR(param);
    invR = inv(R);

    % predict
    M = invF'*Y_km1km1*invF;
    C = M*inv(M + invQ);
    L = eye(param.Nq) - C;
    Y_kkm1 = L*M*L' + C*invQ*C';
    y_kkm1 = L*inv(F)'*y_km1km1;

    % innovate
    sum_I = zeros(size(invR));
    sum_i = zeros(size(z_kj(:,1)));
    for j = 1:size(z_kj,2)
        I = H(:,:,j)'*invR'*H(:,:,j);
        i = H(:,:,j)'*invR'*z_kj(:,j);
        sum_I = sum_I + I;
        sum_i = sum_i + i;
    end
    Y_kk = Y_kkm1 + sum_I;
    y_kk = y_kkm1 + sum_i;
    P_k = inv(Y_kk);
    x_k = P_k*y_kk;

    % update agents
    for i = 1:param.Ni
        agents(i).Q = x_k;
        agents(i).P = P_k;
%         agents(i).alpha = Alpha(Alpha ~= 0);        
    end


end