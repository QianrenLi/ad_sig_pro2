addpath(genpath("SOFTX-D-20-00010-master\"))
%%
clear;
clc;
load("data.mat")

train_features = [];
size_list = [1    33];
for index = 1:4147
    ecg = train_datas(index,:)';
    ecg_signal = ecg(:,1);
    ecg_filtered_isoline = ecg_signal;
    Fs = 300;
    [FPT_MultiChannel,FPT_Cell]=Annotate_ECG_Multi(ecg_filtered_isoline,Fs);
    
    addpath(genpath("E:\Language_lib\Matlab\ad_pro_2"))
    % extract FPTs for Channel 1 (Lead I):
    FPT_LeadI = FPT_Cell{1,1};
    
    Pwave_samples = reshape(FPT_LeadI(:,1:3), [1,size(FPT_LeadI(:,1:3),1)*size(FPT_LeadI(:,1:3),2)]);
    QRS_samples = reshape([FPT_LeadI(:,4),FPT_LeadI(:,6), FPT_LeadI(:,8)] , [1,size(FPT_LeadI(:,1:3),1)*size(FPT_LeadI(:,1:3),2)]);
    Twave_samples = reshape(FPT_LeadI(:,10:12), [1,size(FPT_LeadI(:,10:12),1)*size(FPT_LeadI(:,10:12),2)]);
    num_Pwave = length(Pwave_samples) / 3;
    num_QRS = length(QRS_samples) / 3;
    num_T = length(Twave_samples) / 3;
%     [wave_duration_mean,wave_duration_var,wave_amplitude_mean,wave_amplitude_var] = wave_feature_decompose(QRS_samples,ecg_filtered_isoline');
    try    
    a = wave_feature_decompose(QRS_samples,ecg_filtered_isoline');
    b = wave_feature_decompose(Pwave_samples,ecg_filtered_isoline');
    c = wave_feature_decompose(Twave_samples,ecg_filtered_isoline');
        catch
        train_labels(index,:)
    end
    temp = [num_Pwave,num_QRS,num_T,a,b,c];
    train_features = [train_features; temp];
 
% visualize points
%     figure; 
%     plot(ecg_filtered_isoline(:,1));
%     hold on; 
%     scatter(Pwave_samples, ecg_filtered_isoline(Pwave_samples,1), 'g', 'filled');
%     scatter(QRS_samples, ecg_filtered_isoline(QRS_samples,1), 'r', 'filled');
%     scatter(Twave_samples, ecg_filtered_isoline(Twave_samples,1), 'b', 'filled');
%     title('Filtered ECG');
%     xlabel('samples'); ylabel('voltage');
%     legend({'ECG signal', 'P wave', 'QRS complex', 'T wave'});
end
test_features = [];
for index = 1:1778
    ecg = test_datas(index,:)';
    ecg_signal = ecg(:,1);
    ecg_filtered_isoline = ecg_signal;
    Fs = 300;
    [FPT_MultiChannel,FPT_Cell]=Annotate_ECG_Multi(ecg_filtered_isoline,Fs);
    
    addpath(genpath("E:\Language_lib\Matlab\ad_pro_2"))
    % extract FPTs for Channel 1 (Lead I):
    FPT_LeadI = FPT_Cell{1,1};
    
    Pwave_samples = reshape(FPT_LeadI(:,1:3), [1,size(FPT_LeadI(:,1:3),1)*size(FPT_LeadI(:,1:3),2)]);
    QRS_samples = reshape([FPT_LeadI(:,4),FPT_LeadI(:,6), FPT_LeadI(:,8)] , [1,size(FPT_LeadI(:,1:3),1)*size(FPT_LeadI(:,1:3),2)]);
    Twave_samples = reshape(FPT_LeadI(:,10:12), [1,size(FPT_LeadI(:,10:12),1)*size(FPT_LeadI(:,10:12),2)]);
    
    num_Pwave = length(Pwave_samples) / 3;
    num_QRS = length(QRS_samples) / 3;
    num_T = length(Twave_samples) / 3;
    try
    a = wave_feature_decompose(QRS_samples,ecg_filtered_isoline');
    b = wave_feature_decompose(Pwave_samples,ecg_filtered_isoline');
    c = wave_feature_decompose(Twave_samples,ecg_filtered_isoline');
    catch
        test_labels(index,:)
    end
    temp = [num_Pwave,num_QRS,num_T,a,b,c];
    test_features = [test_features; temp];
end

%% Model training

% save("feature.mat","train_features","test_features") % Feature vector save

Weights = train_labels;
Weights(Weights == 0) = Weights(Weights == 0) + 5;
% Weights = Weights / 11;
disp('Start training.')

% model = fitcsvm(train_features,train_labels, 'KernelFunction','gaussian','KernelScale','auto',...
%     'Standardize',true,'OptimizeHyperparameters','auto',Weights=Weights'); % RBF_kernel

model = fitcsvm(train_features,train_labels, 'KernelFunction','linear','KernelScale','auto',...
    'Standardize',true,Weights=Weights'); 


predict_label = predict(model,test_features);
% [predict_labels, accuracy, dec_values] = svmpredict(test_labels,test_features, model,'-b 1 -q 1');

[confusion_mat,order] = confusionmat(test_labels,predict_label);

%% Result Display
figure; confusionchart(confusion_mat,order);
title("相关矩阵")
disp('Finished.')
M = confusion_mat;
TPR = M(2,2) / (M(2,1) + M(2,2)); 
TNR = M(1,1) / (M(1,1) + M(1,2)); 

M = M';
precision = diag(M)./(sum(M,2) + 0.0001);  
recall = diag(M)./(sum(M,1)+0.0001)'; 
precision = mean(precision);
recall = mean(recall);
score = 2*precision*recall/(precision + recall);


%%
rmpath(genpath("SOFTX-D-20-00010-master\"))
