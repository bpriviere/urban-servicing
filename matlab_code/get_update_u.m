function u_vec = get_update_u(param, agents, info_vec, A, i)

    try 
        consensus_term = 0;
%         for j = 1:param.Ni
%             consensus_term = consensus_term + ...
%                 A(i,j)*(agents(j).u - agents(i).u);
%         end
%         u_vec = info_vec - agents(i).info_vec + consensus_term;
        
%         for j = 1:param.Ni
%             consensus_term = consensus_term + ...
%                 A(i,j)*(agents(j).info_vec - agents(i).info_vec);
%         end
%         u_vec = info_vec + consensus_term;

        for j = 1:param.Ni
            consensus_term = consensus_term + ...
                A(i,j)*(agents(j).u - agents(i).u);
        end
        u_vec = info_vec + consensus_term;

    catch
        u_vec = info_vec;
    end

end