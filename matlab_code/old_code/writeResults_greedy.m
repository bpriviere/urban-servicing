function writeResults_greedy(param, sim_results)


    write_filename_r = param.greedy_output_r_filename;
    dlmwrite(write_filename_r, sim_results.R)

end
