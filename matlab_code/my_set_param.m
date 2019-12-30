function param = my_set_param()


param.fix_on = 1;
param.real_data_on = 0; % chicago taxis or gridworld 
param.delta_des = 30;
param.gamma = 0.7;    % discount factor
param.lambda = 0.8;   % time decay 
param.force_alpha = 0.3; 

param.n_episodes = 5;

param.macro_k_on = 0;
param.CASES_k = 1;
param.CASES_Ni = 1;
param.CASES_Nt = 1;

% param.CASES_k  = [1, 50];
% param.CASES_Nt = [10, 100, 200, 500]; 
% param.CASES_Ni = [10,40,80,100];

param.Nr = round(max(param.CASES_Ni)*1.5);

if param.real_data_on
    param.input_data_filename = '../input/may_1_2017';
    param.training_data_filename = '../input/april_30_2017/23_30_4_2017.csv';
    param.mapfile = '../input/map.shp';
    param.map = shaperead(param.mapfile);
    param.Ns_map = size(param.map,1);
    param.face_color = [1 1 1];
    param.n_state_thresh = 10;
    param.R_comm = 3500; % meters
    param.R_task = param.R_comm/2; 
    param.std_cutoff = 0.5;
    param.scale = 111111; % degrees lat/long to m
    param.C1 = 0.001; % average cost per time, [usd/sec], C1 scales inversely with Reward
    param.C2 = 0.0001; % average taxi speed [degrees/sec], approximately 10m/s
    param.Nq = param.n_state_thresh^3;
else
    param.input_data_filename = '../input/gridWorld_online_data.csv';
    param.training_data_filename = '../input/gridWorld_training_data.csv';
    param.Nx = 3; % number x cells in map
    param.Ny = 2; % number y cells in map
    param.Ns = param.Nx*param.Ny;
    param.R0 = 10;    % initial r value
    param.A0 = 3;    % amplitude of oscillations in reward model
    param.tau_R = 2; % number of periods in reward model  
    param.scale = 1;
    param.R_comm = 0.6; %10; %param.Nx*.4; % meters
%     param.R_task = param.Nx;
    param.R_task = param.R_comm/2;
    param.C1 = 1; % average cost per time, used in ttc, C1 scales inversely with Reward
    param.C2 = 1; % average speed, used in tts, C2 scales directly with Reward  
    param.Nq = param.Ns^3; % 
    param.delta_env = 1;
end

param.task_convergence = 10; % BLLL task convergence criteria
param.tau = 1; % exploration/exploitation constant
param.Q0 = 20;   % initial q value
param.P0 = 0.01;  % initial covariance 
param.NoiseRatio = 0.5; % processnoise/measurementnoise
param.MeasurementNoise = 1; 
param.ProcessNoise = param.NoiseRatio*param.MeasurementNoise;
param.movie_filename = '../results/sim_t';

% plotting
param.agent_marker = 'bo';
param.request_marker = 'k^';
param.request_marker_dropoff = 'ks';

param.trace_color_c_wu  = [0.4660, 0.6740, 0.1880]; %[0.5,0.5,0.5];
param.trace_color_c_wou = [0.4940, 0.1840, 0.5560];
param.trace_color_d_wu  = [0, 0.4470, 0.7410];
param.trace_color_d_wou = 	[0.9290, 0.6940, 0.1250];

param.markersize = 10;
param.fontsize = 20;
param.linewidth = 2;

% initialize here, updated in real time
% param.delta_z_max = 1;
param.underbar_alpha = 1;
param.underbar_lambda2L = 1;
param.last_update_k = 1; 

% param.max_R_online = 1;
% param.min_R_online = 1;
% param = reset_error_bounds(param);
end