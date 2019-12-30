
function agents = add_Request_j_to_Agent_i(i,j,requests,agents)
    agents(i).nr = agents(i).nr + 1;
    agents(i).requests(agents(i).nr).i = requests(j).idx;
     
%     agents(i).requests(agents(i).nr).idx = requests(j).idx;
%     agents(i).requests(agents(i).nr).pickup_x = requests(j).pickup_x;
%     agents(i).requests(agents(i).nr).pickup_y = requests(j).pickup_y;
%     agents(i).requests(agents(i).nr).dropoff_x = requests(j).dropoff_x;
%     agents(i).requests(agents(i).nr).dropoff_y = requests(j).dropoff_y;   
%     agents(i).requests(agents(i).nr).revenue = requests(j).revenue;
%     agents(i).requests(agents(i).nr).ttc = requests(j).ttc;
end