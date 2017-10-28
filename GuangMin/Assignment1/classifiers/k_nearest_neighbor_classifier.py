#coding:utf-8
import numpy as np
class KNearestNeighbor(object):
    def __init__(self):
        pass
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    def predict(self, X, k=1, num_loops=0):
        dists = self.compute_distance(X)
        return self.predict_labels(dists, k=k)

    def compute_distance(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # reshape for broadcasting,remember this way
        dists += np.sum(self.X_train ** 2, axis=1).reshape(1, num_train)
        dists += np.sum(X ** 2, axis=1).reshape(num_test, 1)
        temp = 2 * np.dot(X, self.X_train.T)
        dists -= temp
        return dists


    def predict_labels(self, dists, k):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            #get k-mindistance`s calss
            closest_y = self.y_train[np.argsort(dists[i])[0:k]]
            #vote
            y_pred[i] = np.bincount(closest_y).argmax()
        return y_pred

