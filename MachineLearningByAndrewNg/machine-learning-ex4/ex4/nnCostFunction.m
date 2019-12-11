function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

sizeOfTheta1 = size(Theta1);			% This is a debug comment 25x401
sizeOfTheta2 = size(Theta2);			% This is a debug comment 10x26
valueOfM = m;					% This is a debug comment 5000
sizeOfY = size(y);				% This is a debug comment 5000x1
valueOfNumLabels = num_labels;			% This is a debug comment 10
valueOfLambda = lambda;				% This is a debug comment 0
sizeOfX = size(X);				% This is a debug comment 5000x400
valueOfInputLayerSize = input_layer_size;	% This is a debug comment 400
valueOfHiddenLayerSize = hidden_layer_size;	% This is a debug comment 25

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

p = zeros(size(X, 1), 1);
sizeOfP = size(p);				% This is a debug comment 5000x1

size(sigmoid([ones(m, 1) X] * Theta1'));	% This is a debug comment 5000x25
h1 = sigmoid([ones(m, 1) X] * Theta1');		% 5000x25
size(sigmoid([ones(m, 1) h1] * Theta2'));	% This is a debug comment 5000x10
h2 = sigmoid([ones(m, 1) h1] * Theta2');	% 5000x10
[dummy, p] = max(h2, [], 2);			
sizeOfP = size(p);				% This is a debug comment 5000x1

yModified = zeros(m, num_labels);		% Create a new matrix with m rows and num_labels cols.
sizeOfYMdofied = size(yModified);		% This is a debug comment 5000x10

% Code to add 1 where the digit has been identified.
for i = 1:m,
	index = y(i, :);
	yModified(i, index) = 1;
end

% Cost function
J = (sum(sum((-yModified .* log(h2)) - ((1 - yModified) .* log(1 - h2))))) / m;

% Cost function with regularization
thetaOneSquared = 0;
thetaTwoSquared = 0;
sizeOfTheta1One = size(Theta1);
sizeOfTheta2Two = size(Theta2);

for j = 1:sizeOfTheta1One(1),
	for k = 2:(sizeOfTheta1One(2)),
		thetaOneSquared = thetaOneSquared + (Theta1(j, k) * Theta1(j, k));
	end
end

for j = 1:sizeOfTheta2Two(1),
	for k = 2:(sizeOfTheta2Two(2)),
		thetaTwoSquared = thetaTwoSquared + (Theta2(j, k) * Theta2(j, k));
	end
end

J = J + ((thetaOneSquared + thetaTwoSquared) * (lambda/(2 * m)));

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first timer

%Delta_2 = zeros(sizeOfTheta2Two - 1)
Delta_1 = 0;
Delta_2 = 0;
x = zeros(1, input_layer_size);
sizeOfx = size(x);				% This is a debug comment: 1x400
sizeOfBigX = size(X);				% This is a debug comment: 5000x400

for t = 1:m,
	yModified = zeros(num_labels, 1);	% Create a new num_labelsx1 matrix

	%=========================Start of step one======================================
	x = X(t, :);				% Extracting the first row from 5000 rows
	sizeOfx = size(x);			% This is a debug comment: 1x400
	x = x';					% Take the prime of x
	sizeOfNewX = size(x);			% This is a debug comment: 400x1
	x = [1; x];				% Add a new row on top of x: 401x1
	sizeOfNewX = size(x);			% This is a debug comment: 401x1
	a_1 = x;				% Assign the input feature to a_1
	sizeOfA1 = size(a_1);			% This is a debug comment 401x1
	z_2 = Theta1 * a_1;			% Calculate z_2 = [25*401]*[401*1]
	sizeOfZ2 = size(z_2);			% This is a debug comment: 25x1
	a_2 = sigmoid(z_2);			% Calculate the sigmoid of z2.
	sizeOfA2 = size(a_2);			% This is a debug comment: 25x1
	a_2 = [1; a_2];				% Add a new row on top of a_2.
	sizeOfA2 = size(a_2);			% This is a debug comment: 26x1
	z_3 = Theta2 * a_2;			% Calculate z_3 = [10x26]*[26x1]
	sizeOfZ3 = size(z_3);			% This is a debug comment: 10x1
	a_3 = sigmoid(z_3);			% Calculate the sigmoid of z3.
	sizeOfA3 = size(a_3);			% This is a debug comment: 10x1
	valueOfA3 = a_3;			% This is a debug comment
	%=========================End of step one========================================

	%=========================Start of step two======================================
	valueOfY = y(t, :);			% Get the value of y
	yModified(valueOfY, :) = 1;		% Recode the y vector
	delta_3 = a_3 - yModified;		% Calculate delta_3
	sizeOfDelta3 = size(delta_3);		% This is a debug comment 10x1
	valueOfDelta3 = delta_3;		% This is a debug comment
	%=========================End of step two========================================

	%=========================Start of step three====================================
	delta_2 = (Theta2)' * delta_3;			% delta_2: [26x10] * [10x1]
	sizeOfdelta_2 = size(delta_2);			% This is a debug comment: 26x1
	delta_2 = delta_2(2:end);			% Remove the top row from delta_2
	sizeOfdelta_2 = size(delta_2);			% This is a debug comment: 25x1
	delta_2 = delta_2 .* sigmoidGradient(z_2);	% Calculate delta_2
	sizeOfdelta_2 = size(delta_2);			% This is a debug comment 25x1
	valueOfdelta_2 = delta_2;			% This is a debug comment
	%=========================End of step three======================================

	%=========================Start of step four=====================================
	Theta1_grad = (Theta1_grad + (delta_2 * a_1'));		% Calculate Theta1_grad
	sizeOfTheta1_grad = size(Theta1_grad);			% This is a debug comment: 25x401
	Theta2_grad = (Theta2_grad + (delta_3 * a_2'));		% Calculate Theta2_grad
	sizeOfTheta2_grad = size(Theta2_grad);			% This is a debug comment: 10x26
	%=========================End of step four=======================================
end

%=========================Start of step five===================================
Theta1_grad = (Theta1_grad / m);
sizeOfTheta1_grad = size(Theta1_grad);			% This is a debug comment
Theta2_grad = (Theta2_grad / m);
sizeOfTheta2_grad = size(Theta2_grad);			% This is a debug comment
%=========================End of step five=====================================

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_modified = Theta1;
Theta2_modified = Theta2;

Theta1_modified(:, 1) = 0;
Theta2_modified(:, 1) = 0;

Theta1_grad = Theta1_grad + ((lambda / m) * Theta1_modified);
Theta2_grad = Theta2_grad + ((lambda / m) * Theta2_modified);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
