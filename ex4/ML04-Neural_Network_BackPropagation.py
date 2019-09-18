import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.io import loadmat
import utils4
import pandas as pd

# # #   1.1 Visualizing the data   # # #
data = loadmat(os.path.join('ex4', 'ex4data1.mat'))
X = data['X']   # 5000 x 4
y = data['y'].ravel()   # 5000 x 1
y[y==10] = 0
m = y.size   # 5000
# Randomly select 100 data points for visualization
rand_indices = np.random.choice(m,100,replace=False)
sel = X[rand_indices,:]
utils4.displayData(sel)
# plt.show()

# # #   1.2 Model Representation   # # #
# a1=[1s,x], z2=theta*a1, a2=[1s,sigmoid(z1)], z3=theta*a2, a3=sigmoid(z3)
# 400 input units(not included bias), 25 hidden layer units(not included bias)
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
# Load pre-trained Theta1 and Theta2
weights = loadmat(os.path.join('ex4', 'ex4weights.mat'))
Theta1 = weights['Theta1']   # 25 x401
Theta2 = weights['Theta2']   # 10 x 26
Theta2 = np.roll(Theta2,1,axis=0)
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

# Sigmoid Gradient for BackPropagation
def sigmoidGradient(z):
    g = np.zeros(z.shape)
    g = utils4.sigmoid(z) * (1-utils4.sigmoid(z))
    return g
# Test
z = np.array([-1,-0.5,0,0.5,1])
g = sigmoidGradient(z)

# # #   1.3 Feedforward and Regularized cost function   # # #
def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda_=0.0):
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size+1)],
                        (hidden_layer_size, (input_layer_size + 1)))
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size+1)):],
                        (num_labels, (hidden_layer_size + 1)))
    m = y.size
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    # X: 5000 x 400, y: 5000 x 1, Theta1: 25 x 401, Theta2: 10 x26
    a1 = np.concatenate([np.ones((m,1)),X], axis=1)   # 5000 x 401
    z2 = a1.dot(Theta1.T)   # 5000 x25
    a2 = np.c_[np.ones((z2.shape[0],1)), utils4.sigmoid(z2)]   # 5000 x26
    z3 = a2.dot(Theta2.T)   # 5000 x 10
    a3 = utils4.sigmoid(z3)   # 5000 x 10
    # Map y to binary representation
    y_b = pd.get_dummies(y)   # 5000 x 10
    # Part 1.1 : Cost Function
    J = (-1/m) * np.sum(np.sum(np.log(a3)*(y_b) + np.log(1-a3)*(1-y_b)))
    RJ = (lambda_/(2*m)) * (np.sum(np.square(Theta1[:,1:])) + np.sum(np.square(Theta2[:,1:])))
    J += RJ

# # #    Part 2: BackPropagation   # # #
    # Set accumulator Delta as 0
    Delta1 = 0
    Delta2 = 0
    # Define nodes error in each output layer
    d3 = a3 - y_b  # 5000 x 10
    d2 = d3.dot(Theta2[:,1:]) * sigmoidGradient(z2)   # 5000 x 25
    # Update accumulator
    Delta2 = d3.T.dot(a2)   # 10 x 26
    Delta1 = d2.T.dot(a1)   # 25 x 401
    Theta1_ = np.c_[np.zeros((Theta1.shape[0],1)),Theta1[:,1:]]
    Theta2_ = np.c_[np.zeros((Theta2.shape[0],1)),Theta2[:,1:]]
    # Regularized gradient
    Theta1_grad = (1/m) * Delta1 + (1/m) * lambda_ * Theta1_
    Theta2_grad = (1/m) * Delta2 + (1/m) * lambda_ * Theta2_
    grad = np.concatenate([Theta1_grad.values.ravel(), Theta2_grad.values.ravel()])
    return J, grad
# Test
lambda_ = 1
J, _ = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda_)

# # #   Part 2.2: Random Initialization   # # #
def randInitializeWeights(L_in,L_out,epsilon_init=0.12):
    W = np.zeros((L_out,1+L_in))
    W = np.random.choice(2,(L_out,1+L_in))
    return W
# Initialize weights
initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels)
initial_nn_params = np.concatenate([initial_Theta1.ravel(),initial_Theta2.ravel()],axis=0)

# # #   2.3 Gradient Checking   # # #
utils4.checkNNGradients(nnCostFunction)
lambda_ = 3
utils4.checkNNGradients(nnCostFunction,lambda_)
debug_J, _ = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda_)

# # #   2.4 Learning parameters using scipy.optimize.minize   # # #
options = {'maxiter': 100}
lambda_ = 1
costFunction = lambda p: nnCostFunction(p, input_layer_size,
                                        hidden_layer_size,
                                        num_labels,
                                        X,y,lambda_)
res = optimize.minimize(costFunction,
                      initial_nn_params,
                      jac=True,
                      method='TNC',
                      options=options)
# # Get params
nn_params = res.x
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))
Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))
# # Get training result
pred = utils4.predict(Theta1,Theta2,X)
print('Training Accuracy: %f' % (np.mean(pred == y) * 100))

# # #   2.5 Visualizing the Hidden Layer
utils4.displayData(Theta1[:,1:])