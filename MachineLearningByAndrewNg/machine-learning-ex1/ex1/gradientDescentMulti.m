function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y)  % number of training examples = 20
J_history = zeros(num_iters, 1);

temp = zeros(length(theta), 1);		% temp is now of dimension 4x1.
b = zeros(length(theta), 1);		% b is now of dimension 4x1.

size(theta); 	% = 4x1
theta;
size(X);		% = 20x4
size(y);		% = 20x1

errorVector = zeros(4, 1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %	
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

	%===================SOLUTION TWO Does not work=============
	%for j = 1:length(theta),
	%	for i = 1:m,
	%		x = X(i, :);	%Extracting a single row. x is now of dimension 1x4.
	%		size(x);	% This is a debug comment.
	%		x(:, j);	% This is a debug comment.
	%		b(j, 1) = (b(j, 1) + ((theta' * x') - y(i, 1)) * X(i, j)); 
	%	end
	%end

	%thetaChange = (alpha * b) / m;
	%theta = theta - thetaChange;
	%=========================================================

	%===================SOLUTION ONE==========================
	h = X * theta;						%4x1
	errorVector = h - y;					%4x1
	thetaChange = ((X' * errorVector) * alpha) / m;
	theta = theta - thetaChange;
	%=========================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
