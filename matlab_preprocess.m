clc; clear; close all;
addpath('./utils');
% LibSVM Installation: https://github.com/cjlin1/libsvm
% Run make.m in libsvm-master/matlab/
addpath './utils/libsvm-master/matlab'; 
%% Load and save data
LenECG = 2400;
train = dir(fullfile(pwd,"data/train/*.mat"));
test = dir(fullfile(pwd,"data/test/*.mat"));
train_data = zeros(numel(train),LenECG); train_labels = zeros(numel(train),1);
test_data = zeros(numel(test),LenECG); test_labels = zeros(numel(test),1);
for ii = 1:1:numel(train)
load(fullfile(train(ii).folder,train(ii).name))
train_data(ii,:) = crop_or_pad(value,LenECG);
train_labels(ii) = label; 
end
for ii = 1:1:numel(test)
load(fullfile(test(ii).folder,test(ii).name))
test_data(ii,:) = crop_or_pad(value,LenECG);
test_labels(ii) = label; 
end
save('data.mat','train_data','train_labels','test_data','test_labels')
