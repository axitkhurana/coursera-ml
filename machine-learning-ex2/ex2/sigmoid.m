function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

[m n] = size(z);
for row = 1:m
    for column = 1:n
        g(row, column) = 1/(1+e^(-z(row, column)));
    endfor
endfor


% =============================================================

end
