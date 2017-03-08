% this function normalizes the data with mean=0 and standard deviation with
% 1
function [norm,mu,sigma]=featureNormalize(data)
X=data(:,2:end);
IDX=data(:,1);
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

norm=[IDX X_norm];
end

% ============================================================


