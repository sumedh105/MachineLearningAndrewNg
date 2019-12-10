function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

valueOfSizeOfZ = size(z);			% This is a debug comment
valueOfG = g;					% This is a debug comment
noOfRows = valueOfSizeOfZ(1); 
noOfCols = valueOfSizeOfZ(2);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

for i = 1: noOfRows,
	for j = 1: noOfCols,
		g(i, j) = g(i, j) + ((sigmoid(z(i, j))) * (1 - (sigmoid(z(i, j)))));
	end
end












% =============================================================




end
