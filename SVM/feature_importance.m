%% Feature evaluation
clear;
clc;
load('feature.mat')
load("data.mat")
%% Eliminate NaN feature


%% Feature normalizing

feature_mean = mean(train_features);

feature_var = sqrt(var(train_features));
train_features = (train_features  - feature_mean)./feature_var;
test_features = (test_features - feature_mean)./feature_var;

%% Ranking Features
% fscchi2
% [idx,scores] = fscchi2(train_features,train_labels);
% fscmrmr
% [idx,scores] = fscmrmr(train_features,train_labels);

% figure
% bar(scores(idx))
% xlabel('Predictor rank')
% ylabel('Predictor importance score')
% idx = idx(1:10);

% fscnca
mdl = fscnca(train_features,train_labels);
[B,I] = sort(mdl.FeatureWeights);
figure
bar(flipud(B))
xticks(1:length(I))
xticklabels(flipud(I))
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',15)
ax= gca;
i= 1:length(find(mdl.FeatureWeights > 0.5)); 
ax.XTickLabel(i)= cellfun(@(s)['\bf ' num2str(s)],ax.XTickLabel(i),'UniformOutput',false);
xlabel('Feature index','FontSize',15,'FontName','Time New Roman')
ylabel('Feature weight','FontSize',15,'FontName','Time New Roman')
title('Feature Importance Measure')
grid

%%
temp = ["SD1","SD2","appen","wave_duration_mean","wave_duration_var",...
    "wave_amplitude_mean","wave_amplitude_var","wave_width_mean","wave_width_var", ...
"wave_half_width_mean_pre","wave_half_width_var_pre","wave_half_width_mean_post","wave_half_width_var_post"];
feature_str = ["num_Pwave","num_QRS","num_T",temp + "_QRS",temp + "_P",temp + "_T"];
% Example: idx = [  4     5     9    12    13    14    15    16    20    21    22    24    26    27    28    29];
idx =  find(mdl.FeatureWeights > 0.5)';
adopted_feature_name = feature_str(idx);

%% Feature reduction
% train_features_eva = train_features(:,idx(1:10));
% test_features_eva = test_features(:,idx(1:10));

train_features_eva = train_features(:,idx);
test_features_eva = test_features(:,idx);

%% Train Model
Weights = train_labels;
% weighting metric
Weights(Weights == 0) = Weights(Weights == 0) + 3;
disp('Start training.')
% Linear model 
% model = fitcsvm(train_features_eva,train_labels, 'KernelFunction','linear','KernelScale','auto',...
%     'Standardize',true,OptimizeHyperparameters='all' , Weights=Weights'); 

% RBF model
% model = fitcsvm(train_features_eva,train_labels, 'KernelFunction','rbf','KernelScale','auto',...
%     'Standardize',true ,OptimizeHyperparameters='all', Weights=Weights'); % RBF_kernel

% Guassian Model
%     BoxConstraint    KernelScale    KernelFunction    PolynomialOrder    Standardize
%     _____________    ___________    ______________    _______________    ___________
% 
%        996.28          27.106          gaussian             NaN             false 
model = fitcsvm(train_features_eva,train_labels, 'KernelFunction','gaussian','KernelScale',27.106,...
    'Standardize',false ,'BoxConstraint',996.28, Weights=Weights'); % Gaussian_kernel

predict_label = predict(model,test_features_eva);
[confusion_mat,order] = confusionmat(test_labels,predict_label);
%% Confusion Matrix
figure; 
labels = categorical(["Arrhythmias";"Normal"]);
confusionchart(confusion_mat,labels);
title("Confustion Matrix")
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