function stacked_results = stack_results( param, sim_results, stacked_results)

    try 
        stacked_results.Q = cat(3, stacked_results.Q, sim_results.Q);
        stacked_results.A = cat(3, stacked_results.A, sim_results.A);        
        stacked_results.R = cat(1, stacked_results.R, sim_results.R);
    catch
        stacked_results.Q = sim_results.Q;
        stacked_results.A = sim_results.A;
        stacked_results.R = sim_results.R;
        stacked_results.data = sim_results.data;
    end
end