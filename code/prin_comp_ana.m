% return both pc_data and PC matrix
function [pc_data, PC]=prin_comp_ana(data)
X=data(:,2:end);
IDX=data(:,1);
[m,n]=size(X);

U=zeros(n);
S=zeros(n);

%computing co-variance matrix 
Sigma=(X'*X)/m;
[U,S,~]=svd(Sigma);

K=1024;
Z=zeros(size(X,1),K);

PC=U(:,1:K);
Z=X*PC;

pc_data=[IDX Z];




