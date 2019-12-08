function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	a = 0;
	b = 0;
	for i = 1:m,
		x = X(i,:);
		a = a + ((theta' * x') - y(i, :));
		b = b + (((theta' * x') - y(i, :)) * x(1, 2));
	end

	temp0 = theta'(1, 1) - alpha * (a / m);
	temp1 = theta'(1, 2) - alpha * (b / m);
	theta(1, :) = temp0;
	theta(2, :) = temp1;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
