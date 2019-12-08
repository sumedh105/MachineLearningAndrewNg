function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
sumedh = size(X);
jambekar = size(y);
w = size(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%================================================================
%a = log(1 / (1 + e.^-(X * theta)));
%b = -y;
%c = log(1 - (1 / (1 + e.^-(X * theta))));
%d = (1 - y);

%J = ((b * a) - (d * c)) / m;
%================================================================

valueOfX = X;			% This is a debug comment
valueOfTheta = theta;		% This is a debug comment
valueOfy = y;			% This is a debug comment

%========================================Calculating cost function===================================================
a = 0;
for i = 1:m,
	x = X(i, :);
	a = a + ((-y(i, :) * log(1/(1 + e^-(x * theta)))) - ((1 - y(i, :)) * log(1 - (1 / (1 + e^-(x * theta))))));
end

J = a / m;
valueOfJ = J;			% This is a debug comment
%=======================================End of cost function calculation=============================================

%========================================Calculating gradient descent===================================================
%b = zeros(size(theta)); 
%for j = 1:length(theta),
%	for i = 1:m,
%		x = X(i, :);
%		b(j, :) = b(j, :) + (1 / (1 + e^-(x * theta))) * X(i, j);
%	end
%end

%grad = b ./ m

hyp = sigmoid(X * theta);
valueOfHyp = hyp;					% This is a debug comment
sizehyp = size(hyp);					% This is a debug comment
sizeY = size(y);					% This is a debug comment
errorVector = (hyp - y);
valueOfErrorVector = errorVector;			% This is a debug comment
sizeErrorVector = size(errorVector);			% This is a debug comment
sizeX = size(X);					% This is a debug comment
grad = (X' * errorVector)./m;
% =======================================End of gradient descent calculation============================================

end
