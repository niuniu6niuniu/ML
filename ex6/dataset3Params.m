function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% X = 211x2; y = 211x1; Xval = 200x2; size(yval) = 200x1

% 8 testing value
testList = [0.01,0.03,0.1,0.3,1,3,10,30];  

% modelTest = zeros(size(CTest,2),size(sigmaTest,2));   % modelTest = 8x8
% predictions = zeros(size(Xval,1),size(CTest,2)*size(sigmaTest,2));   % predictions = 200x64
% errorTest = zeros(1,size(CTest,2)*size(sigmaTest,2));   % errorTest = 64 real numbers  
minErr = 1.0;   % Starting standard condition
for i = 1:length(testList)   % For C
  for j = 1:length(testList)   % For sigma
    modelTest = svmTrain(X, y, testList(i), @(x1, x2) gaussianKernel(x1, x2, testList(j)));
    predictions = svmPredict(modelTest,Xval);   % predictions = 200x1
    errorTest = mean(double(predictions ~= yval));   % errorTest = 1 numbers
    if errorTest < minErr
      C = testList(i);
      sigma = testList(j);
      minErr = errorTest;
    end
  end
end

##[val p] = min(errorTest);
##a = idivide(p, size(sigmaTest,2),'ceil');
##C = CTest(a);
##b = mod(p, size(sigmaTest,2));   % optimal in sigmaTest
##sigma = sigmaTest(b);

% =========================================================================

end
