function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%% Compute the cost and gradient of regularized linear 
%  regression for a particular choice of theta.
%
h     = X * theta; % m x 2 * 2 x 1 = m x 1
h_y   = h - y; % m x 1
theta = [0; theta(2:end)];
J     = (1/(2*m)) * sum(h_y.^2) + (lambda/(2*m)) * sum(theta.^2);
% =========================================================================

grad = (1/m) .* (X' * h_y) + (lambda/m).*theta; % 2 x m * m x 1 = 2 x 1, theta = 2 x 1

grad = grad(:);

end
