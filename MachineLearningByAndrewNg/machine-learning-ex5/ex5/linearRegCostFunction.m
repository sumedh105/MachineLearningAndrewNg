function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

sizeOfX = size(X);				% This is a debug comment, 12x2
sizeOfY = size(y);				% This is a debug comment, 12x1
valueOfY = y;					% This is a debug comment
sizeOfTheta = size(theta);			% This is a debug comment, 2x1
valueOfLambda = lambda;				% This is a debug comment
sizeofGrad = size(grad);			% This is a debug comment, 2x1
valueOfM = m;					% This is a debug comment

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%==============================Calculate cost function================================================
hypothesis = ((X * theta) - y);				% Calculate the hypothtesis, 12x1
sqError = sum(hypothesis .^ 2);				% Find the sum of the squared error
J = (sqError / (2 * m));				% Find the cost function

thetaModified = zeros(size(theta));			% New variable theta modified
thetaModified = theta;					% Initialize theta modified with theta	
thetaModified(1, 1) = 0;				% Make the first col as zero
regularizedVar = sum(thetaModified' .^ 2);		% Find the regularized variable
regularizedVar = (regularizedVar * (lambda / (2 * m)));
J = J + regularizedVar;					% Cost function with regularized parameter
%==============================End of cost function calculation=======================================


%==============================Calculate gradient=====================================================

grad = ((1 / m) * (X' * (hypothesis))) + (lambda / m) * thetaModified; % Calculate grad

%==============================End of gradient calculation============================================

% =========================================================================

grad = grad(:);

end
