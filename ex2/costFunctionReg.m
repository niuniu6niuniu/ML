function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m
  z = X * theta;
  h(i) = sigmoid(z(i));
end

J = (-1/m) * (log(h) * y + log(1-h) * (1-y)) + (lambda/(2*m)) * (sum(theta.^2) - theta(1)^2) ;

grad(1) = (1/m) *  ((h - y') * X(:,1))';

for j = 2:length(theta)
  grad(j) = (1/m) *  ((h - y') * X(:,j))' + (lambda/m) * theta(j);
end

% mask(1) = 0;
% grad = (1/m) *  ((h - y') * X)' + (lambda/m) * (theta .* mask);

% =============================================================

end
