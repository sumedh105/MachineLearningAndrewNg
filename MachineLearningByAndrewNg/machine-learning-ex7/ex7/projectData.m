function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

sizeOfX = size(X);				% This is a debug comment: 50x2
sizeOfU = size(U);				% This is a debug comment: 2x2
sizeOfK = size(K);				% This is a debug comment: 1x1
sizeOfZ = size(Z);				% This is a debug comment: 50x1
valueOfK = K;					% This is a debug comment: 1

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%
U_reduce = U(:, 1:K);
sizeOfUreduce = size(U_reduce);			% This is a debug comment: 2x1
Z = X * U_reduce;

% =============================================================

end
