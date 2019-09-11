import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

# Task 1: Identity matrix
A = np.eye(5)

# Task 2: Data visualization
data = np.loadtxt(os.path.join('ex1','ex1data1.txt'), delimiter=',')
X = np.c_[np.ones(data.shape[0]),data[:,0]]   # X = data[:,0] X = np.stack([np.ones(m), X], axis=1)
y = np.c_[data[:,1]]                          # y = data[:,1]
# m training samples
m = y.size
def plotData(x,y):
    plt.plot(x, y, 'ro', ms=5, mec='k')
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
# plt.show()

# Task 3: Cost function
theta1 = np.array([[0],[0]])   # theta1 = [[0],[0]]
theta2 = np.array([[-1],[2]])  # theta2 = [[-1],[2]]
J = 0
def computeCost(X, y, theta):
    h = X.dot(theta)
    J = 1 / (2 * m) * np.sum(np.square(h - y))
    return J

# Task 4: Gradient descent
def gradientDescent(X, y, theta, alpha, num_iters):
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        h = X.dot(theta)
        theta = theta - alpha*(1/m)*(X.T.dot(h-y))
        J_history.append(computeCost(X, y, theta))
    return theta, J_history
theta = np.array([[0],[0]])
iterations = 1500
alpha = 0.01
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
# Data Visualization
# plotData(X[:,1],y)
# plt.plot(X[:,1], np.dot(X, theta), '-')
# plt.legend(['Training data', 'Linear regression'])
# plt.show()

# Task 5: Make prediction on population size of 35,000 and 70,000
predict1 = np.dot([1,3.5], theta)
predict2 = np.dot([1,7.0], theta)

# Task 6: Compare gradient descent with scikit-learn
# xx = np.arange(5,23)
# yy = theta[0] + theta[1] * xx
# plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
# plt.plot(xx, yy, label='Linear regression-Gradient descent')
# regr = linear_model.LinearRegression()
# regr.fit(X[:,1].reshape(-1,1), y.ravel())
# plt.plot(xx, regr.intercept_+regr.coef_*xx,label='Scikit-Learn')
# plt.xlim(4,24)
# plt.xlabel('Population of City in 10,000s')
# plt.ylabel('Profit in $10,000s')
# plt.legend(loc=4)
# plt.show()

# Task 7: Surface plot & Contour plot
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i,j] = computeCost(X, y, [theta0,theta1])
J_vals = J_vals.T
# Surface plot
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Surface')
# Contour plot
ax = plt.subplot(122)
plt.contour(theta0_vals,theta1_vals,J_vals,linewidths=2,cmap='viridis',levels=15)
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
plt.title('Contour, showing minimumn')
# plt.show()