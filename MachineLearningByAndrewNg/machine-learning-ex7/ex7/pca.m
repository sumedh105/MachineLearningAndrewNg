function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

sizeOfX = size(X);				% This is a debug comment: 50x2
sizeOfU = size(U);				% This is a debug comment: 2x2
sizeOfA = size(S);				% This is a debug comment: 2x2
valueOfM = m;					% This is a debug comment: 50
valueOfN = n;					% This is a debug comment: 2

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

sigma = (X' * X) / m;				% Code to calculate the covariance matrix

[U, S, V] = svd(sigma);

% =========================================================================

end
