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

print('compare1:')
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad)

# do the gradient check once again with regularization turned on
# you didn't forget the regularization gradient did you?
print('compare2:')
loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad)


