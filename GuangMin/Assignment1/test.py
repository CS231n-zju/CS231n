#coding:utf-8
from Assignment1.load import load_CIFAR10
cifar10_dir = 'I:\Workplace\PycharmProjects\cs231n\Assignment1\cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)