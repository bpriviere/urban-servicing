
% test script for data

clear all; close all; clc;

param = my_set_param();
data = readCityData_v2(param, param.input_data_dir);

