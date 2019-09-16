import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import utils

# Load data
data = np.loadtxt(os.path.join('ex2', 'ex2data1.txt'), delimiter=',')
X = data[:,0:2]
y = data[:,2]

# Visualize data
pos = y == 1
neg = y == 0
def plotData(X,y):
    fig = plt.figure()
    plt.plot(X[pos,0], X[pos,1], 'k+', lw=2, ms=10)
    plt.plot(X[neg,0], X[neg,1], 'ko', mfc='y', ms=8, mec='k', mew=1)
plotData(X,y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not Admitted'])
#plt.show()
pass

# Task 1: Sigmoid Function
def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)
    g = 1/(1+np.exp(-z))
    return g

# Setup matrix appropriately, add ones to X for intercept
m, n = X.shape
X = np.concatenate([np.ones((m,1)), X], axis=1)

# Task 2: Cost Function & Gradient Descent
def costFunction(theta,X,y):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(X.dot(theta))
    J = (-1/m) * (np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y))
    grad = (1/m) * (X.T.dot(h-y))
    return J,grad
# Do some test
initial_theta = np.zeros(n+1)
# test_theta = np.array([-24,0.2,0.2])
# cost, grad = costFunction(test_theta,X,y)

# Task 4: Learning parameters using scipy.optimize
options = {'maxiter': 400}
res = optimize.minimize(costFunction,
                        initial_theta,
                        (X, y),
                        jac=True,
                        method='TNC',
                        options=options)
# cost function at optimized theta in the fun property
cost = res.fun
# the optimized theta is in the x property
theta = res.x
# Visualize data
utils.plotDecisionBoundary(plotData,theta,X,y)

# Task 5: Predict data
def predict(theta,X,threshold=0.5):
    m = X.shape[0]
    p = np.zeros(m)
    p = sigmoid(X.dot(theta)) >= threshold
    return p.astype('int')

# Predict probability for a student with score 45 on exam 1 and score 85 on exam 2
prob = sigmoid(np.array([1,45,85]).dot(theta))
# print(prob)

# Computr accuracy on training set
p = predict(theta,X)
# print('Training Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))