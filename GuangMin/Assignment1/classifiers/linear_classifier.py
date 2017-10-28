from Assignment1.classifiers.linear_svm import svm_loss_vectorized
from Assignment1.classifiers.softmax import softmax_loss_vectorized

import numpy as np

class LinearClassifier(object):
    def __init__(self, *args, **kwargs):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d : loss %f' % (it, num_iters, loss))
        return loss_history

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        y_pred = np.argmax(np.dot(X, self.W), axis=1)

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        pass


class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
