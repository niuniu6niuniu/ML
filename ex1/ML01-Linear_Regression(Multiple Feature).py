import numpy as np
from matplotlib import pyplot as plt
import os

# Load data
data = np.loadtxt(os.path.join('ex1', 'ex1data2.txt'), delimiter=',')
X = data[:,:2]
y = data[:,2]
m = y.size

# Task 1: Feature Normalization
def featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    for i in range(len(X[0])):
        mu[i] = np.mean(X[:,i])
        sigma[i] = np.std(X[:,i])
        X_norm[:,i] = [(x - mu[i])/sigma[i] for x in X_norm[:,i]]
    return X_norm, mu, sigma
X_norm, mu, sigma = featureNormalize(X)

# Add intercept to X: (47,3)
X = np.concatenate([np.ones((m,1)), X_norm], axis=1)


# Task 2: Compute Cost
def computeCostMulti(X,y,theta):
    m = y.shape[0]
    J = 0
    h = X.dot(theta)   # (47,1)
    J = 1/(2*m) * (h-y).T.dot((h-y))
    return J


# Task 3: Gradient Descent
def gradientDescentMulti(X,y,theta,alpha,num_iters):
    m = y.shape[0]
    J_history = []
    for i in range(num_iters):
        h = X.dot(theta)   # (47,1)
        theta -= alpha * (1/m) * X.T.dot((h-y))
        J_history.append(computeCostMulti(X,y,theta))
    return theta, J_history

alpha = 0.1
num_iters = 400
theta = np.zeros(3)   # (3,1)
theta, J_history = gradientDescentMulti(X,y,theta,alpha,num_iters)

plt.plot(np.arange(len(J_history)), J_history, lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
# plt.show()
predict = np.array([[1650,3]])
predict = (predict - mu) / sigma
predict = np.concatenate([np.ones((1,1)), predict], axis=1)
print(predict.dot(theta))

