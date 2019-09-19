import os
import numpy as np
import re
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from IPython.display import HTML, display, clear_output
try:
    plt.rcParams["animation.html"] = "jshtml"
except ValueError:
    plt.rcParams["animation.html"] = "html5"
from scipy import optimize
from scipy.io import loadmat
import utils7

# # #   Part 1: K-means clustering   # # #
# # #   Part 1.1: Implementing K-means
# # #   Part 1.1.1: Finding closest centroids
def findClosestCentroids(X, centroids):
    # Set K clusters/centroids
    K = centroids.shape[0]
    # The index of clusters that each node belongs to
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        dis = K * [0]
        for j in range(K):
            dis[j] = (X[i,:]-centroids[j,:]).dot((X[i,:]-centroids[j,:]).T)
        idx[i] = dis.index(min(dis))
    return idx
# Test
# Load an example dataset that we will be using
# data = loadmat(os.path.join('ex7', 'ex7data2.mat'))
# X = data['X']   # 300 x 2
# # Select an initial set of centroids
# K = 3   # 3 Centroids
# initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
# # Find the closest centroids for the examples using the initial_centroids
# idx = findClosestCentroids(X, initial_centroids)
# print(idx.shape)
# print('Closest centroids for the first 3 examples:')
# print(idx[:3])
# print('(the closest centroids should be 0, 2, 1 respectively)')

# # #   Part 1.1.2: Computing centroid means
# Compute average centroid for each clusters
def computeCentroids(X, idx, K):
    # Useful variables
    m, n = X.shape   # 300 x 2
    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))
    for i in range(K):
        todo = np.zeros((1,n))
        count = 0
        for j in range(m):
            if idx[j] == i:
                todo += X[j,:]
                count += 1
        centroids[i,:] = (1/count) * todo
    return centroids
# Test
# centroids = computeCentroids(X, idx, K)
# print('Centroids computed after initial finding of closest centroids:')
# print(centroids)
# print('\nThe centroids should be')
# print('   [ 2.428301 3.157924 ]')
# print('   [ 5.813503 2.633656 ]')
# print('   [ 7.119387 3.616684 ]')

# # #   Part 1.2: K-means on example dataset

# Load an example dataset
# data = loadmat(os.path.join('ex7', 'ex7data2.mat'))
# Settings for running K-Means
# K = 3
# max_iters = 10
# initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
# # Run K-Means algorithm. The 'true' at the end tells our function to plot the progress of K-Means
# centroids, idx, anim = utils7.runkMeans(X, initial_centroids,
#                                        findClosestCentroids,
#                                         computeCentroids,
#                                         max_iters, True)
# anim
# plt.show()

# # #   Part 1.3 Random initialization
def kMeansInitCentroids(X, K):
    m, n = X.shape   # 300 x 2
    # You should return this values correctly
    centroids = np.zeros((K, n))
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]
    return centroids

# # #   Part 1.4 Image compression with K-means

# Load 128x128 color image (bird_small.png)
# img = mpl.image.imread(os.path.join('ex7', 'bird_small.png'))

# ======= Experiment with these parameters ================
# You should try different values for those parameters
# K = 16
# max_iters = 10
#
# # Load an image of a bird
# # Change the file name and path to experiment with your own images
# A = mpl.image.imread(os.path.join('ex7', 'bird_small.png'))
# # ==========================================================
#
# # Divide by 255 so that all values are in the range 0 - 1
# A /= 255
#
# # Reshape the image into an Nx3 matrix where N = number of pixels.
# # Each row will contain the Red, Green and Blue pixel values
# # This gives us our dataset matrix X that we will use K-Means on.
# X = A.reshape(-1, 3)
#
# # When using K-Means, it is important to randomly initialize centroids
# initial_centroids = kMeansInitCentroids(X, K)
#
# # Run K-Means
# centroids, idx = utils7.runkMeans(X, initial_centroids,
#                                  findClosestCentroids,
#                                  computeCentroids,
#                                  max_iters)
#
# # We can now recover the image from the indices (idx) by mapping each pixel
# # (specified by its index in idx) to the centroid value
# # Reshape the recovered image into proper dimensions
# X_recovered = centroids[idx, :].reshape(A.shape)

# Display the original image, rescale back by 255
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# ax[0].imshow(A*255)
# ax[0].set_title('Original')
# ax[0].grid(False)
#
# # Display compressed image, rescale back by 255
# ax[1].imshow(X_recovered*255)
# ax[1].set_title('Compressed, with %d colors' % K)
# ax[1].grid(False)
# plt.show()

# # #   Part: 2 Principal Component Analysis   # # #

# 2.1 Example Dataset
# Load the dataset into the variable X
data = loadmat(os.path.join('ex7', 'ex7data1.mat'))
X = data['X']

#  Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bo', ms=10, mec='k', mew=1)
plt.axis([0.5, 6.5, 2, 8])
plt.gca().set_aspect('equal')
plt.grid(False)

# # #   2.2 Implementing PCA
# Step 1: Compute the covariance matrix of the data.
# Step 2: Use SVD (in python we use numpy's implementation np.linalg.svd)
# to compute the eigenvectors. These will correspond to the
# principal components of variation in the data.
def pca(X):
    # Useful values
    m, n = X.shape
    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)
    Sigma = (1/m) * X.T.dot(X)
    U, S, V = np.linalg.svd(Sigma)
    return U, S
# Test
#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = utils7.featureNormalize(X)
#  Run PCA
U, S = pca(X_norm)
#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], 'bo', ms=10, mec='k', mew=0.25)

for i in range(2):
    ax.arrow(mu[0], mu[1], 1.5 * S[i]*U[0, i], 1.5 * S[i]*U[1, i],
             head_width=0.25, head_length=0.2, fc='k', ec='k', lw=2, zorder=1000)

ax.axis([0.5, 6.5, 2, 8])
ax.set_aspect('equal')
ax.grid(False)

print('Top eigenvector: U[:, 0] = [{:.6f} {:.6f}]'.format(U[0, 0], U[1, 0]))
print(' (you should expect to see [-0.707107 -0.707107])')

# # #   Part 2.3 Dimensionality Reduction with PCA
# # # 2.3.1 Projecting the data onto the principal components
def projectData(X, U, K):
    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))
    U_reduce = U[:,:K]
    Z = X.dot(U_reduce)
    return Z
# Test
#  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
# print('Projection of the first example: {:.6f}'.format(Z[0, 0]))
# print('(this value should be about    : 1.481274)')

# # #   Part  2.3.2 Reconstructing an approximation of the data
def recoverData(Z, U, K):
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    U_reduce = U[:,:K]
    X_rec = Z.dot(U_reduce.T)
    return X_rec

X_rec  = recoverData(Z, U, K)
print('Approximation of the first example: [{:.6f} {:.6f}]'.format(X_rec[0, 0], X_rec[0, 1]))
print('       (this value should be about  [-1.047419 -1.047419])')

#  Plot the normalized dataset (returned from featureNormalize)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(X_norm[:, 0], X_norm[:, 1], 'bo', ms=8, mec='b', mew=0.5)
ax.set_aspect('equal')
ax.grid(False)
plt.axis([-3, 2.75, -3, 2.75])

# Draw lines connecting the projected points to the original points
ax.plot(X_rec[:, 0], X_rec[:, 1], 'ro', mec='r', mew=2, mfc='none')
for xnorm, xrec in zip(X_norm, X_rec):
    ax.plot([xnorm[0], xrec[0]], [xnorm[1], xrec[1]], '--k', lw=1)
# plt.show()

# # #   Part 2.4 Face Image Dataset
#  Load Face dataset
data = loadmat(os.path.join('ex7', 'ex7faces.mat'))
X = data['X']

#  Display the first 100 faces in the dataset
utils7.displayData(X[:100, :], figsize=(8, 8))

#  normalize X by subtracting the mean value from each feature
X_norm, mu, sigma = utils7.featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

#  Visualize the top 36 eigenvectors found
utils7.displayData(U[:, :36].T, figsize=(8, 8))
# plt.show()

# # #   Part 2.4.2 Dimensionality Reduction
#  Project images to the eigen space using the top k eigenvectors
#  If you are applying a machine learning algorithm
K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a shape of: ', Z.shape)
#  Project images to the eigen space using the top K eigen vectors and
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed
K = 100
X_rec  = recoverData(Z, U, K)

# Display normalized data
utils7.displayData(X_norm[:100, :], figsize=(6, 6))
plt.gcf().suptitle('Original faces')

# Display reconstructed data from only k eigenfaces
utils7.displayData(X_rec[:100, :], figsize=(6, 6))
plt.gcf().suptitle('Recovered faces')
pass

# # #   Part 2.5: PCA for visualization
# this allows to have interactive plot to rotate the 3-D plot
# The double identical statement is on purpose
# see: https://stackoverflow.com/questions/43545050/using-matplotlib-notebook-after-matplotlib-inline-in-jupyter-notebook-doesnt
%matplotlib notebook
%matplotlib notebook
from matplotlib import pyplot


A = mpl.image.imread(os.path.join('ex7', 'bird_small.png'))
A /= 255
X = A.reshape(-1, 3)

# perform the K-means clustering again here
K = 16
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = utils7.runkMeans(X, initial_centroids,
                                 findClosestCentroids,
                                 computeCentroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
sel = np.random.choice(X.shape[0], size=1000)

fig = pyplot.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], cmap='rainbow', c=idx[sel], s=8**2)
ax.set_title('Pixel dataset plotted in 3D.\nColor shows centroid memberships')
pass


# Subtract the mean to use PCA
X_norm, mu, sigma = utils7.featureNormalize(X)

# PCA and project the data to 2D
U, S = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Reset matplotlib to non-interactive
%matplotlib inline

fig = pyplot.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

ax.scatter(Z[sel, 0], Z[sel, 1], cmap='rainbow', c=idx[sel], s=64)
ax.set_title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
ax.grid(False)
pass


