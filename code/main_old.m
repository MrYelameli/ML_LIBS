%% This is the old code, written in 2015, Here direct the data is divided 
% after normalization and after pre-processing(PCA). 
% In this code the inputs are direct PCA_train and PCA_test data. 

tic;
clc;
clear all;
close all;


data=dlmread('PCA_trD.txt');
NTrain = size(data,1);
train_data=data(:,2:end);
train_label=data(:,1);
%[train_label,d]=sortrows(train_label);

dataTest=dlmread('PCA_teD.txt');
test_data=dataTest(:,2:end);
test_label=dataTest(:,1);
%[test_label,~]=sortrows(test_label); 


%combine the data together just to fit format.
totalData = [train_data; test_data];
totalLabel = [train_label; test_label];



[N D] = size(totalData);
labelList = unique(totalLabel(:));
NClass = length(labelList);

% #######################
% Determine the train and test index
% #######################

trainIndex = zeros(N,1); trainIndex(1:NTrain) = 1;
testIndex = zeros(N,1); testIndex( (NTrain+1):N) = 1;
trainData = totalData(trainIndex==1,:);
trainLabel = totalLabel(trainIndex==1,:);
testData = totalData(testIndex==1,:);
testLabel = totalLabel(testIndex==1,:); 

% #######################
% Parameter selection using 3-fold cross validation
% #######################

bestcv = 0;

log2c =1.5;
%log2g =0.000976;
dimen=375;

bestcv=zeros(length(log2c),length(dimen));

cv=zeros(length(log2c),length(dimen));
for i=1:length(log2c)
    %for k =1:length(log2g)
    for j=1:length(dimen)
        
        %for linear
         cmd=['-q -c ',num2str(log2c(i)), ' -t 0'];
        
        %for RBF
        %cmd=['-q -c ',num2str(log2c(i)),' -g ',num2str(log2g)];
        trD=trainData(:,1:dimen(j));
        cv(i,j) = get_cv_ac(trainLabel, trD, cmd, 3);
        if (cv(i,j) >= bestcv),
            bestcv = cv(i,j); bestc = log2c(i) ; bestDimen=dimen(j); %bestg=log2g(k);
        end
        
        
        %for Linear
         fprintf('%g %g %g (best c=%g, rate=%g, bestDimen=%g)\n',log2c(i),cv(i,j),dimen(j),bestc,bestcv,bestDimen);
        %for RBF
        %fprintf('%g %g %g %g(best c=%g, best g=%g rate=%g, bestDimen=%g)\n',log2c(i),log2g(k),cv(i,j),dimen(j),bestc,bestcv,bestDimen)
        
    end
 end





%   figure;
%   plot(log2c,dimen,cv,'--gs','LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','b');
%   xlabel('C');
%   ylabel('Dimension');
%   zlabel('Accuracy');
%   legend('Linear SVM');
%   title('PCA SVM LinearKernel');


% #######################
% Train the SVM in one-vs-rest (OVR) mode
% #######################

%for RBF
%bestParam = ['-q -c ', num2str(bestc), ' -g ', num2str(bestg)];
TRD=trainData(:,1:bestDimen);

%for linear 
bestParam = ['-q -c ',num2str(bestc), ' -t 0'];

model = ovrtrain(trainLabel, TRD, bestParam);

% #######################
% Classify samples using OVR model
% #######################
[predict_label, accuracy, prob_values] = ovrpredict(testLabel, testData, model);
fprintf('Accuracy = %g%%\n', accuracy * 100);

%%%% PLOT CONFUSION MATRIX %%%%
 %%one hot encoding%%
output=zeros(size(predict_label,1),10);
 
 for i=1:10
     rows = predict_label==i;
     output(rows,i)=1;
 end
output=output';

 target=zeros(size(testLabel,1),10);
 
 for i=1:10
     rows=testLabel==i;
     target(rows,i)=1;
 end
 target=target';
 
plotconfusion(target,output)
toc;



