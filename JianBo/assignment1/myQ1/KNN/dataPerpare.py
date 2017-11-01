#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wjbKimberly on 17-10-31
import numpy as np
import sys
sys.path.append("../../../")
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

def dataPrepare(cifar10_dir,num_training,num_test):
    # Load the raw CIFAR-10 data.

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # As a sanity check, we print out the size of the training and test data.
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    #print
    # Training data shape:  (50000, 32, 32, 3)
    # Training labels shape:  (50000,)
    # Test data shape:  (10000, 32, 32, 3)
    # Test labels shape:  (10000,)

    # Visualize some examples from the dataset.
    # We show a few examples of training images from each class.
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    # plt.show()


    # Subsample the data for more efficient code execution in this exercise
    # Only choose top 5000 in training data
    # Only choose top 500 in test data

    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]


    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    return X_train, y_train, X_test, y_test
