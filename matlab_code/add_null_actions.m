function [agents, requests] = add_null_actions(param, agents, requests)

    tf = 0;
    for j = 1:param.Nr
        if tf > requests(j).t
            tf = t;
        end
    end

    for i = 1:param.Ni
        
        j = param.Nr + i;
        requests(j).idx = j;
        requests(j).pickup_x = agents(i).x;
        requests(j).pickup_y = agents(i).y;
        requests(j).dropoff_x = agents(i).x;
        requests(j).dropoff_y = agents(i).y;     
        requests(j).revenue = 0;
        requests(j).ttc = 1;
        requests(j).t = tf;
        
        agents(i).nr = agents(i).nr + 1;
        agents(i).requests(agents(i).nr).i = j;
    end

end