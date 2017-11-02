#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wjbKimberly on 17-11-2


# First implement the naive softmax loss function with nested loops.
# Open the file cs231n/classifiers/softmax.py and implement the
# softmax_loss_naive function.
import random
import numpy as np
import sys
sys.path.append("../../")
from cs231n.classifiers.softmax import softmax_loss_naive
from dataPrepare import get_CIFAR10_data
import time
import matplotlib.pyplot as plt



# Invoke the above function to get our data.
cifar10_dir = '../../cs231n/datasets/cifar-10-batches-py'
num_training=49000
num_validation=1000
num_test=1000
num_dev=500
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data(cifar10_dir,
                                                                                num_training,
                                                                                num_validation,
                                                                                num_test,
                                                                                num_dev)
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print('dev data shape: ', X_dev.shape)
print('dev labels shape: ', y_dev.shape)


# Generate a random softmax weight matrix and use it to compute the loss.
W = np.random.randn(3073, 10) * 0.0001
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

# As a rough sanity check, our loss should be something close to -log(0.1).
print('loss: %f' % loss)
print('sanity check: %f' % (-np.log(0.1)))


# # Complete the implementation of softmax_loss_naive and implement a (naive)
# # version of the gradient that uses nested loops.
# loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)
#
# # As we did for the SVM, use numeric gradient checking as a debugging tool.
# # The numeric gradient should be close to the analytic gradient.
# from cs231n.gradient_check import grad_check_sparse
# f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
# grad_numerical = grad_check_sparse(f, W, grad, 10)
#
# # similar to SVM case, do another gradient check with regularization
# loss, grad = softmax_loss_naive(W, X_dev, y_dev, 5e1)
# f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 5e1)[0]
# grad_numerical = grad_check_sparse(f, W, grad, 10)


# Now that we have a naive implementation of the softmax loss function and its gradient,
# implement a vectorized version in softmax_loss_vectorized.
# The two versions should compute the same results, but the vectorized version should be
# much faster.
tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

from cs231n.classifiers.softmax import softmax_loss_vectorized
tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# As we did for the SVM, we use the Frobenius norm to compare the two versions
# of the gradient.
grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
print('Gradient difference: %f' % grad_difference)



# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set.
from cs231n.classifiers import Softmax

results = {}
best_val = -1
best_softmax = None
learning_rates = [1e-7, 5e-7]
regularization_strengths = [2.5e4, 5e4]

for li in learning_rates:
    for rsi in regularization_strengths:
        softmax = Softmax()
        loss_hist = softmax.train(X_train, y_train, learning_rate=li, reg=rsi,
                              num_iters=1500, verbose=True)
        y_train_pred = softmax.predict(X_train)
        y_train_acc = np.mean(y_train == y_train_pred)
        y_val_pred = softmax.predict(X_val)
        y_val_acc = np.mean(y_val == y_val_pred)
        print('learning rate is %.10f and regularization strength is %.10f' % (li, rsi))
        print('validation accuracy: %f \n' % (y_val_acc,))
        results[(li, rsi)] = (y_train_acc, y_val_acc)
        if best_val < y_val_acc:
            best_softmax = softmax
            best_val = y_val_acc
################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained softmax classifer in best_softmax.                          #
################################################################################
#pass
################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)



# evaluate on test set
# Evaluate the best softmax on test set
y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))



# Visualize the learned weights for each class
w = best_softmax.W[:-1, :]  # strip out the bias
w = w.reshape(32, 32, 3, 10)

w_min, w_max = np.min(w), np.max(w)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)

    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])