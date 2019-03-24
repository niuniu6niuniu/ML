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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hx = X * theta;      % hx = 12 X 1
% regularization without theta0
reg1 = (lambda/(2*m)) * (theta' * theta - theta(1)^2);   
J = 1/(2*m) * ((hx - y)' * (hx - y)) + reg1 ;   % J = 1 x 1

% create a mask for regularization of theta without theta0
mask = ones(size(theta));   
mask(1) = 0;  
reg2 = (lambda/m) * (theta .* mask);
grad = (1/m) * (X' * (hx - y)) + reg2;   % grad = 2 x 1

% =========================================================================

grad = grad(:);

end
