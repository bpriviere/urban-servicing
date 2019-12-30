function agents = add_Agent_j_to_Agent_i(i,j,agents)
    try 
        agents(i).nn = agents(i).nn + 1;
    catch
        agents(i).nn = 1;
    end
    agents(i).neighbors(agents(i).nn).i = agents(j).i;
end