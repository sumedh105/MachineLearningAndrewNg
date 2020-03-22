function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

sizeOfx1Before = size(x1);		% This is a debug comment: 1x3
sizeOfX2Before = size(x2);		% This is a debug comment: 1x3

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

sizeOfx1 = size(x1);			% This is a debug comment: 3x1
sizeOfx2 = size(x2);			% This is a debug comment: 3x1
valueOfSigma = sigma;			% This is a debug comment: 2

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%
A = 0;
for k = 1:length(x1),
	A = A + ((x1(k, :) - x2(k, :)) ^ 2);
end

sim = exp(-(A / (2 * (sigma ^ 2))))


% =============================================================
    
end
