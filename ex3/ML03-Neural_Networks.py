import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.io import loadmat
import utils3

# # #   2.1 Load data   # # #

# Input images: 20 x 20, 10 labels: 1 to 10
data = loadmat(os.path.join('ex3','ex3data1.mat'))
X = data['X']   # 5000 x 400
y = data['y'].ravel()   # 5000 x 1
# Set label 10 to label 0
y[y==10] = 0
m = y.size

# Visualize data
indice = np.random.permutation(m)
rand_indices = np.random.choice(m,100,replace=False)
sel = X[rand_indices, :]
utils3.displayData(sel)
# plt.show()

# Neural Network parameters
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
# Load pre-trained weights
weights = loadmat(os.path.join('ex3', 'ex3weights.mat'))
Theta1 = weights['Theta1']   # 25 x 401
Theta2 = weights['Theta2']   # 10 x 26
# Swap first column and last column of Theta2, due to legacy from MATLAB
Theta2 = np.roll(Theta2, 1, axis=0)

# # #   2.2 Forward Propagation and Prediction   # # #
def predict(Theta1,Theta2,X):
    # Make sure the input is 2-d
    if X.ndim == 1:
        X = X[None]
    # Sample size & categories
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    p = np.zeros(X.shape[0])
    # X: 5000 x 401
    a1 = np.concatenate([np.ones((m,1)), X], axis=1)   # 5000 x 401
    z2 = a1.dot(Theta1.T)   # 5000 x 25
    a2 = np.c_[np.ones((z2.shape[0],1)), utils3.sigmoid(z2)] # 5000 x 26
    z3 = a2.dot(Theta2.T)   # 5000 x 10
    a3 = utils3.sigmoid(z3)
    p = np.argmax(a3, axis=1)
    return p
pred = predict(Theta1,Theta2,X)
print('Training Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))

