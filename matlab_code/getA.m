function A = getA(param, agents)

    A = zeros(param.Ni);
    for i = 1:param.Ni
        di = agents(i).nn;
        for j = 1:agents(i).nn
            k = agents(i).neighbors(j).i;
            if i ~= k
                dk = agents(k).nn;
                A(i,k) = 1/max( [ di,dk]);
            end
        end
    end
    
    for i = 1:param.Ni
        A(i,i) = 1- sum(A(:,i));
    end
end