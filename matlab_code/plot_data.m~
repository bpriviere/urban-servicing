

figure()

data = param.data;
R = zeros(size(data,1), param.Ns*param.Ns);
TTC = zeros(size(data,1), param.Ns*param.Ns);
T = zeros(size(data,1), param.Ns*param.Ns);
for i = 1:size(data,1)
    SP = loc_to_state( param, param.data(i,1), param.data(i,2));
    SD = loc_to_state( param, param.data(i,3), param.data(i,4));
    A_idx = (SP-1)*param.Ns + SD;
    
    if and( (SP == SD) , SP == 1)
        'wait';
    end
    
    R(i,A_idx) = data(i,5);
    TTC(i, A_idx) = data(i,6);
    T(i,A_idx) = data(i,7);
    
end

figure()
title('Reward for Actions')
for i = 1:param.Ns*param.Ns
    subplot(param.Ns, param.Ns, i)
    
    sp = floor(i/param.Ns);
    sd = mod(i-1, param.Ns)+1;
    ylabel(sprintf('a(%d, %d)', sp,sd));
    xlabel('time')
    plot( T(:,i), R(:,i),'.')
    
end

figure()
title('Reward at State, Actions')
for i = 1:param.Ns^3
    [s_curr,s_p,s_d] = q_to_sa(param, i);
    
    a_idx = (s_p-1)*param.Ns + s_d;
    
    reward
    
    subplot(param.Ns^2, param.Ns, i)
    plot( T(:,a_idx), R(:,q_idx),'.')
    
    
    
end
    