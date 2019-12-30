function H = getH(param,z_kj)
    H = zeros(param.Nq, param.Nq, size(z_kj,2));
    for j = 1:size(z_kj,2)
        H(:,:,j) = eye(param.Nq);
    end
    
%     for j = 1:size(z_kj,2)
%         hj = 1*(z_kj(:,j) ~= 0);
%         H(:,:,j) = diag(hj);
%     end
end