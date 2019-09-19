import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.io import loadmat
import utils6
# Import regular expressions to process emails
import re


# # #   Part 1.1: Data1 Visualization   # # #
# You will have X, y as keys in the dict data
# data = loadmat(os.path.join('ex6', 'ex6data1.mat'))
# X, y = data['X'], data['y'][:, 0]
# Plot training data
# utils6.plotData(X, y)
# plt.show()
# Test different values for C
# C = 1
# model = utils6.svmTrain(X, y, C, utils6.linearKernel, 1e-3, 20)
# utils6.visualizeBoundaryLinear(X, y, model)
# plt.show()

# # #   Part 1.2: SVM with Gaussian Kernels   # # #
def gaussianKernel(x1, x2, sigma):
    sim = 0
    sim = np.exp(-(x1-x2).T.dot(x1-x2) / (2 * (sigma ** 2)))
    return sim
# Test
# x1 = np.array([1, 2, 1])
# x2 = np.array([0, 4, -1])
# sigma = 2
# sim = gaussianKernel(x1, x2, sigma)

# # #  Part 1.1.2: Data2 Visualization   # # #
# Load from ex6data2
# You will have X, y as keys in the dict data
# data = loadmat(os.path.join('ex6', 'ex6data2.mat'))
# X, y = data['X'], data['y'][:, 0]
# # Plot training data
# utils6.plotData(X, y)
# SVM Parameters
# C = 1
# sigma = 0.1
# model= utils6.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
# utils6.visualizeBoundary(X, y, model)
# plt.show()

# # #  Part 1.1.3: Data3 Visualization   # # #
# Load from ex6data3
# You will have X, y, Xval, yval as keys in the dict data
# data = loadmat(os.path.join('ex6', 'ex6data3.mat'))
# X, y, Xval, yval = data['X'], data['y'][:, 0], data['Xval'], data['yval'][:, 0]
# # Plot training data
# utils6.plotData(X, y)
# plt.show()

# Finding best values for C and sigma
def dataset3Params(X, y, Xval, yval):
    C = 1
    sigma = 0.3
    minError = 1.0
    candidates = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for i in range(len(candidates)):
        C_ = candidates[i]
        for j in range(len(candidates)):
            sigma_ = candidates[j]
            model = utils6.svmTrain(X, y, C_, gaussianKernel, 1e-3, 20)
            predictions = utils6.svmPredict(model, X)
            error = np.mean(predictions != yval)
            if error < minError:
                C = C_
                sigma = sigma_
                minError = error
    return C, sigma

# Try different SVM Parameters here
# C, sigma = dataset3Params(X, y, Xval, yval)   # C = 1, sigma = 0.3
# C = 1
# sigma = 0.3
# # Train the SVM
# # model = utils.svmTrain(X, y, C, lambda x1, x2: gaussianKernel(x1, x2, sigma))
# model = utils6.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
# utils6.visualizeBoundary(X, y, model)
# plt.show()

# # #   Part 2: Spam Classification   # # #

# 2.1 Pre-processing Emails
# - Lower-casing
# - Stripping HTML
# - Normalizing URLs
# - Normalizing Email Addresses
# - Normalizing Numbers
# - Normalizing Dollars
# - Word Stemming
# - Removal of non-words

# Part 2.1: Map words in email to indices in vocabulary list
def processEmail(email_contents, verbose=True):
    # Load vocabulary list
    vocabList = utils6.getVocabList()
    word_indices = []
    # - Lower case
    email_contents = email_contents.lower()
    # - Strip all HTML
    email_contents = re.compile('<[^<>]+>').sub(' ', email_contents)
    # - Handle Numbers
    email_contents = re.compile('[0-9]+').sub('number', email_contents)
    # - Handle URLS
    email_contents = re.compile('(http|https)://[^\s]*').sub('httpaddr', email_contents)
    # - Handle Email Addresses
    email_contents = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)
    # - Handle $ sign
    email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)
    # - Remove any empty word string
    email_contents = [word for word in email_contents if len(word) > 0]
    # - Stem the email contents word by word
    stemmer = utils6.PorterStemmer()
    processed_email = []
    for word in email_contents:
        # Remove any remaining non alphanumeric characters in word
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)
        processed_email.append(word)
        if len(word) < 1:
            continue
    # Word look up
    for word in processed_email:
        try:
            word_indices.append(vocabList.index(word))
        except:
            pass
    if verbose:
        print('----------------')
        print('Processed email:')
        print('----------------')
        print(' '.join(processed_email))
    return word_indices

# Extract Feature
with open(os.path.join('ex6', 'emailSample1.txt')) as fid:
    file_contents = fid.read()

word_indices  = processEmail(file_contents)
# Print Stats
print('-------------')
print('Word Indices:')
print('-------------')
print(word_indices)

# # #   Part 2.2: Construct feature vector for emails
def emailFeatures(word_indices):
    n = 1899
    x = np.zeros((n,1))
    for i in range(len(word_indices)):
        x[word_indices[i]] = 1
    return x
features = emailFeatures(word_indices)
print(len(features))

# # #   Part 3: Train Linear SVM for Spam Classification   # # #
# Load spam email samples
data = loadmat(os.path.join('ex6', 'spamTrain.mat'))
X, y = data['X'], data['y'][:, 0]
C = 0.1
model = utils6.svmTrain(X, y, C, utils6.linearKernel, 1e-3, 20)
p = utils6.svmPredict(model, X)
print('Training Accuracy: %f' % (np.mean(p == y) * 100))

# # #   Part 4: Test Spam Classification   # # #
Xtest, ytest = data['Xtest'], data['ytest'][:, 0]
p_test = utils6.svmPredict(model, Xtest)
print('Training Accuracy: %f' % (np.mean(p == y) * 100))
