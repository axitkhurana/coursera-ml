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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

X = [ones(m, 1) X];
z_2 = X * Theta1';  % 5000 x 25
a_2_no_1 = sigmoid(z_2);

m_2 = size(a_2_no_1, 1);
a_2 = [ones(m_2, 1) a_2_no_1];

z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);  % 5000 x 10
h_x = a_3;

h_X_pos = 0;
h_X_neg = 0;
for ex = 1:m
    for label = 1:num_labels
        if (label == y(ex))
            h_X_pos = h_X_pos + (-1 * log(h_x(ex, label)));
        else
            h_X_neg = h_X_neg + (-1 * log(1 - h_x(ex, label)));
        end
    end
end
J = (1/m * (sum(h_X_pos) + sum(h_X_neg)));
J_extra = lambda/(2*m) * (sum(sum(Theta1(:, 2:end) .^2 , 2)) + sum(sum(Theta2(:, 2:end) .^2 , 2)));
J = J + J_extra;

%  ----------------------------

Y = zeros(size(y, 1), num_labels);
x = 1;
for i = y'
    Y(x, i) = 1;
    x = x + 1;
end

del_3 = a_3 - Y; %  5000 x 10
del_2 =  (del_3 * Theta2) .* sigmoidGradient([ones(size(z_2, 1), 1) z_2]);
del_2 = del_2(:, 2:end);

% del_2 5000 x 25: 5000 x 10 * 10 x 25 .* 5000 x 25
Theta1_grad = del_2' * X;  % 25 x 401
Theta2_grad = del_3' * a_2; % 10 x 26


Theta1_reg = Theta1;
Theta1_reg(:, 1) = zeros(size(Theta1_reg, 1), 1);
Theta1_grad = 1/m * (Theta1_grad + lambda * Theta1_reg);
% Theta1_grad 26x401 Theta_reg 25x401


Theta2_reg = Theta2;
Theta2_reg(:, 1) = zeros(size(Theta2_reg, 1), 1);
Theta2_grad = 1/m * (Theta2_grad + lambda *  Theta2_reg);


% recode y to Y
% I = eye(num_labels);
% Y1 = zeros(m, num_labels);
% for i=1:m
%   Y1(i, :)= I(y(i), :);
% end
% 
% if Y == Y1
%     fprintf('Y is same')
% else
%     fprintf('Y is same')
% end
% 
% feedforward
% a1 = [ones(m, 1) X];
% z2 = a1*Theta1';
% a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
% z3 = a2*Theta2';
% a3 = sigmoid(z3);
% h = a3;
% 
% % calculte penalty
% p = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2));
% 
% % calculate J
% J = sum(sum((-Y).*log(h) - (1-Y).*log(1-h), 2))/m + lambda*p/(2*m);
% 
% % calculate sigmas
% sigma3 = a3.-Y;
% sigma2 = (sigma3*Theta2).*sigmoidGradient([ones(size(z2, 1), 1) z2]);
% sigma2 = sigma2(:, 2:end);
% 
% % accumulate gradients
% delta_1 = (sigma2'*a1);
% delta_2 = (sigma3'*a2);
% 
% % calculate regularized gradient
% p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
% p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
% Theta1_grad = delta_1./m + p1;
% Theta2_grad = delta_2./m + p2;
% 


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
