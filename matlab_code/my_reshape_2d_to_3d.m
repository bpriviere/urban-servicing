function q_3d = my_reshape_2d_to_3d(param, q_2d)
    q_3d = zeros(param.Nq, param.Ni, param.Nt);
    for i = 1:param.Ni
        start_idx = (i-1)*param.Nq + 1;
        end_idx = start_idx + param.Nq - 1;
        temp = q_2d(start_idx:end_idx,:);
        q_3d(:,i,:) = temp;
    end
end