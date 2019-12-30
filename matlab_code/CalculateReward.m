function [agents, param] = CalculateReward(param, agents, requests)
    
    used_requests = zeros(param.Ni,1);
    for i = 1:param.Ni
        if any(used_requests == agents(i).curr_action_i)
            agents(i).reward = 0;
        else
            a = requests(agents(i).curr_action_i);
            s = loc_to_state(param, agents(i).x, agents(i).y);
            tts = calcTTS(param,s,a);
            agents(i).reward =  a.revenue - param.C1*(tts + a.ttc);
            
%             if agents(i).reward < param.min_R_online
%                 param.min_R_online = agents(i).reward;
%             end
%             
%             if agents(i).reward > param.max_R_online
%                 param.max_R_online = agents(i).reward;
%             end
            
            used_requests(i) = agents(i).curr_action_i;
        end
    end
end