function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

degree = 6;
out = ones(size(X1(:,1))); % out首列为全1
for i = 1:degree % out第i列是个i+1次式子，由对应的i个X1(i), X2(i)相乘而得
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j); % (:, end+1)表示在末尾添加一列
    end
end

end