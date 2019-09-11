import numpy as np
import os

# Load data
data = np.loadtxt(os.path.join('ex1', 'ex1data2.txt'), delimiter=',')
X = data[:,:2]
y = data[:,2]
m = y.size
X = np.concatenate([np.ones((m,1)), X], axis=1)
theta = np.zeros(X.shape[1])

# Normal Equations
def normalEquations(X,y):
    theta = np.zeros(X.shape[1])
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta
theta = normalEquations(X,y)
predict = np.array([[1,1650,3]])
print(predict.dot(theta))
