#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wjbKimberly on 17-10-31

from dataPrepare import dataPrepare
import numpy as np
import sys
sys.path.append("../../")
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.gradient_check import grad_check_sparse
import time
from cs231n.classifiers.linear_svm import svm_loss_vectorized
import matplotlib.pyplot as plt
from cs231n.classifiers import LinearSVM


cifar10_dir = '../../cs231n/datasets/cifar-10-batches-py'
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

X_train, X_val, X_test, X_dev,y_train, y_val, y_test, y_dev=dataPrepare(cifar10_dir,num_training,num_validation,num_test,num_dev)

#SVM Classifier
#Your code for this section will all be written inside cs231n/classifiers/linear_svm.py.
#As you can see, we have prefilled the function compute_loss_naive which uses for loops to evaluate the multiclass SVM loss function.

# Evaluate the naive implementation of the loss we provided for you:
# generate a random SVM weight matrix of small numbers
W = np.random.randn(3073, 10) * 0.0001
loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005)
#print('initialize W randomly: ',W)
print('loss: %f' % (loss, ))


# Once you've implemented the gradient, recompute it with the code below
# and gradient check it with the function we provided for you

# Compute the loss and its gradient at W.
print('Compute the loss and its gradient at W.')
loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)

# Numerically compute the gradient along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should match
# almost exactly along all dimensions.


# print('compare1:')
# f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
# grad_numerical = grad_check_sparse(f, W, grad)
#
# # do the gradient check once again with regularization turned on
# # you didn't forget the regularization gradient did you?
# print('compare2:')
# loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
# f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]
# grad_numerical = grad_check_sparse(f, W, grad)


# Next implement the function svm_loss_vectorized; for now only compute the loss;
# we will implement the gradient in a moment.
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))
# print('Naive gradient: ',grad_naive)

tic = time.time()
loss_vectorized, g2 = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))
# print('Vectorized gradient:' ,g2)

# The losses should match but your vectorized implementation should be much faster.
print('difference: %f' % (loss_naive - loss_vectorized))


# Complete the implementation of svm_loss_vectorized, and compute the gradient
# of the loss function in a vectorized way.

# The naive implementation and the vectorized implementation should match, but
# the vectorized version should still be much faster.
tic = time.time()
_, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Naive loss and gradient: computed in %fs' % (toc - tic))

tic = time.time()
_, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Vectorized loss and gradient: computed in %fs' % (toc - tic))

# The loss is a single number, so it is easy to compare the values computed
# by the two implementations. The gradient on the other hand is a matrix, so
# we use the Frobenius norm to compare them.
difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('difference: %f' % difference)


# In the file linear_classifier.py, implement SGD in the function
# LinearClassifier.train() and then run it with the code below.
#
svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                      num_iters=1500, verbose=True)
toc = time.time()
print('That took %fs' % (toc - tic))


# A useful debugging strategy is to plot the loss as a function of
# iteration number:
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()


# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
y_train_pred = svm.predict(X_train)
print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
y_val_pred = svm.predict(X_val)
print('validation accuracy: %f \n' % (np.mean(y_val == y_val_pred), ))

# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of about 0.4 on the validation set.
learning_rates = [1e-8,1e-7,5e-5,1e-4]
regularization_strengths = [2.5e4, 5e4]

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1  # The highest validation accuracy that we have seen so far.
best_svm = None  # The LinearSVM object that achieved the highest validation rate.

################################################################################
# TODO:                                                                        #
# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a linear SVM on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the LinearSVM object that achieves this  #
# accuracy in best_svm.                                                        #
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the SVMs don't take much time to train; once you are #
# confident that your validation code works, you should rerun the validation   #
# code with a larger value for num_iters.                                      #
################################################################################
#pass

for li in learning_rates:
    for rsi in regularization_strengths:
        svm = LinearSVM()
        loss_hist = svm.train(X_train, y_train, learning_rate=li, reg=rsi,
                          num_iters=1500, verbose=True)
        y_train_pred = svm.predict(X_train)
        y_train_acc = np.mean(y_train == y_train_pred)
        y_val_pred = svm.predict(X_val)
        y_val_acc = np.mean(y_val == y_val_pred)
        print('learning rate is %.10f and regularization strength is %.10f'% (li,rsi) )
        print('validation accuracy: %f \n' % (y_val_acc,))
        results[(li,rsi)]=(y_train_acc,y_val_acc)
        if best_val<y_val_acc:
            best_svm=svm
            best_val=y_val_acc

################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)


# Visualize the cross-validation results
import math
x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.subplot(2, 1, 1)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')

# plot validation accuracy
colors = [results[x][1] for x in results] # default size of markers is 20
plt.subplot(2, 1, 2)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.show()


# Evaluate the best svm on test set
y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)


# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength, these may
# or may not be nice to look at.
w = best_svm.W[:-1, :]  # strip out the bias
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