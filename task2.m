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
% save('data.mat','train_data','train_labels','test_data','test_labels')
%% Feature Extraction
% 参考: https://github.com/Aiwiscal/ECG-ML-DL-Algorithm-Matlab
% ToDoList:
% R波检测算法/QRS  
% heartbeat        
% 小波变换
Ntrain = 500; % size(train_data,1)
Ntest = 100; % size(test_data,1)
Nfeature = 10;
train_features = zeros(Ntrain,Nfeature); train_labels = train_labels(1:Ntrain);
test_features = zeros(Ntest,Nfeature); test_labels = test_labels(1:Ntest);
for ii = 1:Ntrain
    [C,L]=wavedec(train_data(ii,:),5,'db6');  %% db6小波5级分解；
    train_features(ii,:) = C(1:Nfeature);
end
for ii = 1:Ntest
    [C,L]=wavedec(test_data(ii,:),5,'db6');  %% db6小波5级分解；
    test_features(ii,:) = C(1:Nfeature);
end
tic
disp('Start training.')
model = svmtrain(train_labels, train_features, '-t 1 -g 0.07 -b 1 -q 1'); % RBF_kernel
[predict_labels, accuracy, dec_values] = svmpredict(test_labels,test_features, model,'-b 1 -q 1');
[confusion_mat,order] = confusionmat(test_labels,predict_labels);
figure; confusionchart(confusion_mat,order);
disp('Finished.')
toc

