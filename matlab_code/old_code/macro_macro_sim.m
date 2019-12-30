

clear all; close all; clc;

rng(4);

my_set_param;


R_CASE = [0.1, 0.5, 1  , 10];
N_CASE = [  1,   5, 10 , 20];
Jstar = zeros(length(R_CASE), length(N_CASE));


param.Nr = max(N_CASE)*2;
if ~param.real_data_on
    generateData(param);
end

for i_rcase = 1:length(R_CASE)
    
    param.R_comm = R_CASE(i_rcase);
    param.R_task = param.R_comm/2;
    
    for i_ncase = 1:length(N_CASE)
        
        param.Ni = N_CASE(i_ncase);
        param.Nr = 2*param.Ni;

        if param.cent_on
        %     clear all; close all; clc;
%             my_set_param;
            param.algo = 0;
            fprintf('Centralized\n')
            main_opt_routing;
        end

        if param.greedy_on
        %     clear all; close all;
%             my_set_param;
            fprintf('Greedy\n')
            greedy_ver;
        end

        if param.dist_ms_on
        %     clear all; close all; 
%             my_set_param;
            param.algo = 1;
            fprintf('DIST MS\n')
            main_opt_routing;
        end

        if param.dist_ss_on
        %     clear all; close all; 
%             my_set_param;
            param.algo = 2;
            fprintf('DIST SS\n')
            main_opt_routing;
        end
        
        reward_cent = dlmread(param.c_output_r_filename);
        reward_dist = dlmread(param.dist_ms_output_r_filename);
        Jstar(i_rcase, i_ncase) = sum(reward_dist)/sum(reward_cent);
        
    end
end
plot_macro_q(Jstar, R_CASE, N_CASE);




% plot_state_space(param, agents, requests);
% plot_macro_r(param);


