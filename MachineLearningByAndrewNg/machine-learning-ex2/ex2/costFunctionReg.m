function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

sizeOfTheta = size(theta);
sizeOfX = size(X);
sizeOfy = size(y);
theta;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

[resultCost, resultGrad] = costFunction(theta, X, y);

a = 0;
b = 0;

for j = 2:length(theta),
	a = a + (theta(j, :) * theta(j, :));
end

b = (lambda / (2 * m)) * a;

%========This is the value of cost function after regularization===========
J = resultCost + b;
%==========================================================================

%========This is the value of gradient descent after regularization========
c = zeros(size(theta));
for j = 2:length(theta),
	c(j, :) = c(j, :) + (theta(j, :) * (lambda/m));
end

grad(1, :) = resultGrad(1, :);
for j = 2: length(theta),
	grad(j, :) = resultGrad(j, :) + c(j, :);
end
grad
% =============================================================

end
