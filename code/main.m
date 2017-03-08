%This code is written for classifyting the rocks types using PCA and SVM
%techniques. The PCA is used to reduce the dimension of the data and then 
%classification was performed using Support Vector Machine algorithm.
%some of the  sub functions are inspired from https://sites.google.com/site/kittipat/home
%The LIBSVM package is used for SVM algorithm http://www.csie.ntu.edu.tw/~cjlin/libsvm/

clear all;
clc;
close all;

%read the file 
%read the train data
tr_data=dlmread('../Data/train_data.txt');
%read the test data
te_data=dlmread('../Data/test_data.txt');

%The next step is to normalize the data, the data is normalized 
norm_tr=featureNormalize(tr_data);
norm_te=featureNormalize(te_data);

%the next step is data reduction using PCA algorithm 

pca_tr,PC=prin_comp_ana(norm_tr);
pca_te=[norm_te(:,1) norm_te(:,2:end)*PC];

NTrain=size(pca_tr,1);
trainData=pca_tr(:,2:end);
trainLabel=pca_tr(:,1);
%[trainLabel,d]=sortrows(trainLabel);

testData=pca_te(:,2:end);
testLabel=pca_te(:,1);
%[testLabel,~]=sortrows(testLabel); 

%Support vector machine, parameter selection using 3-fold cross validation

bestcv = 0;

log2c =1:10:20 ; 
log2g =0.1:0.1:0.2;
dimen=25:25:50;
bestcv=zeros(length(log2c),1);

cv=zeros(length(log2c),length(dimen),length(log2g));
for i=1:length(log2c)
    for k =1:length(log2g)
    for j=1:length(dimen)
        
        
        % for RBF
        cmd = ['-q -c ', num2str(log2c(i)), ' -g ', num2str(log2g)];
        
        trD=trainData(:,1:dimen(j));
        cv(i,j) = get_cv_ac(trainLabel, trD, cmd, 3);
        if (cv(i,j) >= bestcv),
            bestcv = cv(i,j); bestc = log2c ; bestDimen=dimen(j); bestg=log2g;
        end
        
        %for RBF
        fprintf('%g %g %g %g (best c=%g, bestg=%g, rate=%g, bestDimen=%g)\n', log2c, log2g, cv(i,j), dimen(j), bestc, bestg, bestcv, bestDimen);
        
        % for Linear
        %fprintf('%g %g %g (best c=%g, rate=%g, bestDimen=%g)\n',log2c(i),cv(i,j),dimen(j),bestc,bestcv,bestDimen);
        
        
    end
    end
end

%After getting best parameters, check the accuracy on test dataset. 
TRD=trainData(:,1:bestDimen);
bestParam = ['-q -c ', num2str(bestc),' -g ', num2str(bestg)];

model = ovrtrain(trainLabel, TRD, bestParam);

[predict_label, accuracy, prob_values] = ovrpredict(testLabel, testData, model);

%this gives the accuracy on test dataset. 
fprintf('Accuracy = %g%%\n', accuracy * 100);
