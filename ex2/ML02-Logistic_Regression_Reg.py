import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import utils

data = np.loadtxt(os.path.join('ex2', 'ex2data2.txt'), delimiter=',')
X = data[:,:2]
y = data[:,2]

# Plot data
pos = y == 1
neg = y == 0
def plotData(X,y):
    fig = plt.figure()
    plt.plot(X[pos, 0], X[pos, 1], 'k+', lw=2, ms=10)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)

# Visualize data
plotData(X,y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'], loc='upper right')
pass

# Feature mapping, 6 degree, already add ones intercept to X
X = utils.mapFeature(X[:,0], X[:,1])

# Sigmoid Function
def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)
    g = 1/(1+np.exp(-z))
    return g

# Task 1: Cost Function and Gradient Descent
def costFunctionReg(theta,X,y,lambda_):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(X.dot(theta))
    J = (-1/m) * (np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y)) + (lambda_/(2*m)) * (theta[1:].T.dot(theta[1:]))
    grad = (1/m) * X.T.dot(h-y)
    return J, grad

# Initialize theta and lambda_
initial_theta = np.zeros(X.shape[1])
lambda_ = 1
cost, grad = costFunctionReg(initial_theta,X,y,lambda_)

# Task 2: Learning parameters using scipy.optimize
options = {'maxiter': 100}
res = optimize.minimize(costFunctionReg,
                        initial_theta,
                        (X,y,lambda_),
                        jac=True,
                        method='TNC',
                        options=options)
cost = res.fun
theta = res.x
# Visualize data
utils.plotDecisionBoundary(plotData,theta,X,y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'])
plt.grid(False)
plt.title('lambda = %0.2f' % lambda_)

# Compute accuracy on training set
def predict(theta,X,threshold=0.5):
    m = X.shape[0]
    p = np.zeros(m)
    p = sigmoid(X.dot(theta)) >= threshold
    return p.astype('int')

p = predict(theta,X)
print('Training Accuracy: %.1f %%' % (np.mean(p == y) * 100))