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
I = [ones(m, 1) X]; % X的每一行表示一个样例, 第一列添加偏置单元. I: mx401
H = [ones(m, 1) sigmoid(I * Theta1')]; % Theta1: 25x401, I* Theta1': mx25
O = sigmoid(H * Theta2'); % H: mx26, Theta2: 10x26, O: mx10

J_unreg = 0; % 误差项
for i = 1:m, % 累加每一个样例对应的误差
  h = O(i,:); % 第i个样例是对应下标数字的概率
  y_recode = (1:num_labels)' == y(i); % 样例标签解析为布尔列向量
  J_unreg += -(log(h) * y_recode + log(1 .- h) * (1 .- y_recode));
end;
J_unreg /= m;

J_reg = 0; % 正则化项
J_reg += sum(sum(Theta1(:, 2:end) .^ 2)); % Theta1平方和, 不计偏置项
J_reg += sum(sum(Theta2(:, 2:end) .^ 2)); % Theta2平方和, 不计偏置项
J_reg = (lambda / (2 * m)) * J_reg;

J = J_unreg + J_reg;

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
%               first time.
%
D1 = 0;
D2 = 0;
for i = 1:m,
  % step 1 计算z2 a2 z3 a3
  % Theta1: 25x401, Theta2: 10x26
  a1 = [1; X(i, :)'];                  % a1: 401x1, **input layer**
  z2 = Theta1 * a1;                    % z2: 25x1
  a2 = [1; sigmoid(z2)];               % a2: 26x1,  **hidden layer**
  z3 = Theta2 * a2;                    % z3: 10x1
  a3 = sigmoid(z3);                    % a3: 10x1,  **output layer**
  
  % step 2 计算δ3 (d for δ, D for Δ)
  d3 = a3 - ((1:num_labels)' == y(i)); % d3: 10x1
  
  % step 3 计算σ2(需丢弃Theta2'的偏置项) 
  % Theta2'(2:end, :): 25x10, d3:10x1, z2:25x1
  d2 = Theta2'(2:end, :) * d3 .* sigmoidGradient(z2); 
  
  % step 4 累计Δ(l) += δ(l+1) * (a(l))'
  D2 += d3 * a2';                      % d3: 10x1, a2': 1x26, D2:10x26
  D1 += d2 * a1';                      % d2: 25x1, a1': 1x401,D1:25x401  
end;
Theta1_grad = 1 / m * D1;
Theta2_grad = 1 / m * D2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

Theta1_grad(:, 2:end) += lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) += lambda / m * Theta2(:, 2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
