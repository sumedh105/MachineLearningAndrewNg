function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

sizeOfX = size(X);				% This is a debug comment: 12x1
sizeOfXPoly = size(X_poly);			% This is a debug comment: 12x8
valueOfP = p;					% This is a debug comment: 8
m = length(X);					% This is a debug comment: 12 

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 
for i = 1:m,
	for j = 1:p,
		X_poly(i, j) = X(i) .^ j;
	end
end


% =========================================================================

end
