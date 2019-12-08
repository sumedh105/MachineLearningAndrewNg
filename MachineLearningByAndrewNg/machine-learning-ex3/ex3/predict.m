function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

valueOfThetaOne = Theta1;		% This is a debug comment
sizeOfThetaOne = size(Theta1);		% This is a debug comment (25x401)
valueOfThetaTwo = Theta2;		% This is a debug comment
sizeOfThetaTwo = size(Theta2);		% This is a debug comment (10x26)
valueOfX = X;				% This is a debug comment
sizeOfX = size(X);			% This is a debug comment
valueOfM = m;				% This is a debug comment
valueOfNumLabels = num_labels;		% This is a debug comment
valueOfP = p;				% This is a debug comment
sizeOfP = size(p);			% This is a debug comment

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%XModified = [ones(size(X, 2), 1), X];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%XModified = X'(:, 1);
%sizeOfXModified = size(XModified)					% This is a debug comment (400x1)
%newXModified = [ones(size(XModified, 2), 1); XModified]; 		% Adding a new row of value 1.
%sizeOfNewXModified = size(newXModified)					% This is a debug comment (401x1)
%hiddenLayer = sigmoid(Theta1*newXModified);				% [25x401]*[401x1]
%valueOfHiddenLayer = hiddenLayer;					% This is a debug comment
%sizeOfHiddenLayer = size(hiddenLayer)					% This is a debug comment (25x1)

%hiddenLayerModified = [ones(size(hiddenLayer, 2), 1); hiddenLayer];	% Adding a new row of value 1
%sizeOfHiddenLayerModified = size(hiddenLayerModified)			% This is a debug comment
%outputLayer = sigmoid(Theta2*hiddenLayerModified);			% [10x26]*[26x1]
%sizeOfOutputLayer = size(outputLayer)					% This is a debug comment
%valueOfOutputLayer = outputLayer;

%p = outputLayer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a = zeros(m, num_labels);							
sizeOfA = size(a);								% 5000x10

for i = 1:m,
	XModified = X(i, :);							% 1x400
	sizeOfXModified = size(XModified);					% This is a debug comment, 1x400
	XModified = XModified';							% 400x1
	sizeOfXModified = size(XModified);					% 400x1
	newXModified = [ones(size(XModified, 2), 1); XModified];		% Add a new row on top, 401x1
	sizeOfNewXModified = size(newXModified);				% 401x1
	hiddenLayer = sigmoid(Theta1*newXModified);				% [25x401]*[401x1] = [25x1]
	sizeOfHiddenLayer = size(hiddenLayer);					% 25x1

	hiddenLayerModified = [ones(size(hiddenLayer, 2), 1); hiddenLayer];	% Add a new row on top, 26x1
	sizeOfHiddenLayerModified = size(hiddenLayerModified);			% 26x1
	outputLayer = sigmoid(Theta2*hiddenLayerModified);			% [10x26]*[26x1] = [10x1]
	sizeOfOutputLayer = size(outputLayer);					% [10x1]

	a(i, :) = outputLayer';							% a(i, :) = [1x10]
	sizeOfAi = size(a(i, :));						% [1x10]
end

sizeOfA = size(a);								% [5000x10]

for i = 1:m,
	[maxPredict, maxIndex] = max(a(i, :), [], 2);				% Take the largest value in every row.
	valueOfMaxIndex = maxIndex;
	p(i, :) = maxIndex;							% Assign the largest value to p.
end




% =========================================================================


end
