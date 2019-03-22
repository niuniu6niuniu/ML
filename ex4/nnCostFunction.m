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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ********************************* PART 1 *************************************
% Input layer 
a1 = [ones(m,1) X];   % 5000x401
% Hidden layer
z2 = a1 * Theta1';    % 5000x25
a2 = [ones(size(z2,1),1) sigmoid(z2)];   % 5000x26
% Output layer
z3 = a2 * Theta2';    % 5000x10
a3 = sigmoid(z3);

K = num_labels;
% Construct vectors of y
%t = zeros(size(y,1),K);   % 5000x10
%for i = 1:size(y,1)
%  t(i,y(i)) = 1;
%end
Y = eye(K)(y,:);

% Cost Function
J = (1/m) * sum(sum((-Y .* log(a3)) - ((1-Y) .* log(1-a3))));

% ***************************** PART 1.4 ***************************************
% Regularize Theta
RTheta1 = Theta1(:,2:end);    % 25x400
RTheta2 = Theta2(:,2:end);    % 10x25

RJ = (lambda/(2*m))  * (sum(sum(RTheta1.^2)) + sum(sum(RTheta2.^2)));
J += RJ;

% ******************************** PART 2 **************************************
% Backpropagation
% Input layer
Delta1 = 0;
Delta2 = 0;
% Iterate for all the samples
for t = 1:m
  % Forward Propagation
  a1 = [1; X(t,:)'];     % At each step extract one sample, a1 = 401x1
  z2 = Theta1 * a1;      % z2 = 25x1
  a2 = [1; sigmoid(z2)]; % a2 = 26x1
  z3 = Theta2 * a2;      % z3 = 10x1
  a3 = sigmoid(z3);      % Output = a3 = 10x1
  % Compute last layer error
  d3 = a3 - Y(t,:)';     % d3 = 10x1
  % Compute the middle layer error
  d2 = (RTheta2' * d3) .* sigmoidGradient(z2);   % d2 = 25x1 without bias
  % Accumulation
  Delta2 += (d3 * a2');    % Delta2 = 10x26
  Delta1 += (d2 * a1');    % Delta1 = 25 x 401 
end
% Unregularized graident
Theta1_grad = (1 / m) * Delta1;   % Theta1_grad = 25x401
Theta2_grad = (1 / m) * Delta2;   % Theta2_grad = 10x26
  
%D1 = 0;
%D2 = 0;
%d3 = a3 - t;   % 5000x10
%d2 = d3 * RTheta2 .* sigmoidGradient(z2);   % 5000x25 
%D2 += (d3' * a2);   % 10x26
%D1 += (d2' * a1);   % 25x401
%Theta1_grad = (1/m) * D1;   % 25x401
%Theta2_grad = (1/m) * D2;   % 10x26

% ********************************* PART 3 ************************************
% Regularized gradient
Theta1_grad(:,2:end) += ((lambda/m) * RTheta1);   % Theta1_grad = 25x400
Theta2_grad(:,2:end) += ((lambda/m) * RTheta2);   % Theta2_grad = 10x25

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end