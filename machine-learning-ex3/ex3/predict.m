function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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
I = [ones(m, 1) X]; % 为输入层添加偏置单元, size(I) = [n, 401]
H = sigmoid(I * Theta1'); % 计算隐层单元, size(H) = [n, 25], 每一行是一个元素的对应隐层值
H = [ones(m, 1) H]; % 添加偏置单元, size(H) = [n, 26]
O = sigmoid(H * Theta2'); % 计算输出层单元, size(O) = [n, 10], O(i, j)表示第i个元素是j的概率
[_, p] = max(O, [], 2); % 见predictOneVsAll.m

% =========================================================================


end
