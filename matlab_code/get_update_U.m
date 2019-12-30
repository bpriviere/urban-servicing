function  U_mat = get_update_U(param, agents, info_mat, A, i)

    try 
        consensus_term = 0;
%         for j = 1:param.Ni
%             consensus_term = consensus_term + ...
%                 A(i,j)*(agents(j).U - agents(i).U);
%         end
%         U_mat = info_mat - agents(i).info_mat + consensus_term;

%         for j = 1:param.Ni
%             consensus_term = consensus_term + ...
%                 A(i,j)*(agents(j).info_mat - agents(i).info_mat);
%         end
%         U_mat = info_mat + consensus_term;

        for j = 1:param.Ni
            consensus_term = consensus_term + ...
                A(i,j)*(agents(j).U - agents(i).U);
        end
        U_mat = info_mat + consensus_term;

    catch
        U_mat = info_mat;
    end

end