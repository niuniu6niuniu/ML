import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.io import loadmat
import utils3

# # #   1.1 Load data   # # #

# Input images: 20 x 20, 10 labels: 1 to 10
input_layer_size = 400
num_labels = 10
data = loadmat(os.path.join('ex3','ex3data1.mat'))
X = data['X']   # 5000 x 400
y = data['y'].ravel()   # 5000 x 1
#y[y==10] = 0   # Set label 10 to label 0
m = y.size

# # #   1.2 Visualize the data   # # #

# Randomly select 100 data points to display
rand_indices = np.random.choice(m,100,replace=False)
sel = X[rand_indices,:]
utils3.displayData(sel)
#plt.show()

# # #   1.3 Vectorizing Logistic Regression   # # #
# Train 10 logistic classifiers

# Test data:
theta_t = np.array([-2,-1,1,2],dtype=float)
X_t = np.concatenate([np.ones((5,1)),np.arange(1,16).reshape(5,3,order='F')/10.0],axis=1)
y_t = np.array([1,0,1,0,1])
lambda_t = 3

# Cost Function
def lrCostFunction(theta,X,y,lambda_):
    m = y.size
    if y.dtype == bool:
        y = y.astype(int)
    J = 0
    grad = np.zeros(theta.shape)
    h = utils3.sigmoid(X.dot(theta))
    J = (-1/m) * (np.log(h).dot(y) + np.log(1-h).dot(1-y)) + (lambda_/(2*m)) * (theta[1:].T.dot(theta[1:]))
    mask = np.r_[0,theta[1:]]
    grad = (1/m) * X.T.dot(h-y) + (lambda_/m) * mask
    return J,grad
J, grad = lrCostFunction(theta_t,X_t,y_t,lambda_t)

# # #   1.4 One-vs-all Classification
def oneVsAll(X,y,num_labels,lambda_):
    m, n = X.shape   # 5000 x 400
    all_theta = np.zeros((num_labels, n+1))   # theta for 10 labels, 10 x 401
    X = np.concatenate([np.ones((m,1)), X], axis=1)   # add 1s, 5000 x 401
    # Train 10 classifiers
    initial_theta = np.zeros((n+1,1))   # 401 x 1
    options = {'maxiter': 50}
    for c in range(1,num_labels+1):
        res = optimize.minimize(lrCostFunction,
                                initial_theta,
                                (X, (y==c), lambda_),
                                jac=True,
                                method='TNC',
                                options=options)
        all_theta[c-1] = res.x
    return all_theta

lambda_ = 0.1
all_theta = oneVsAll(X,y,num_labels,lambda_)   # 10 x 401

# # #   1.5 Predict one-vs-all
def predictOneVsAll(all_theta,X):
    m = X.shape[0]
    p = np.zeros(m)   # 5000 x 1
    X = np.concatenate([np.ones((m,1)), X], axis=1)   # 5000 x 401
    prob = X.dot(all_theta.T)   # 5000 x 10
    p = np.argmax(prob, axis=1) + 1
    return p

pred = predictOneVsAll(all_theta,X)
print('Training Set Accuracy: {:.2f} %'.format(np.mean(pred == y) * 100))
