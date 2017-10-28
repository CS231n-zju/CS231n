# coding:utf-8
import numpy as np

def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    N, C = X.shape[0], W.shape[1]
    for i in range(N):
        f = np.dot(X[i], W)
        f -= np.max(f)
        loss = loss + np.log(np.sum(np.exp(f))) - f[y[i]]
        dW[:, y[i]] -= X[i]
        s = np.exp(f).sum()
        for j in range(C):
            dW[:, j] += np.exp(f[j]) / s * X[i]

    loss = loss / N + 0.5 * reg * np.sum(W * W)
    dW = dW / N + reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    N = X.shape[0]
    f = np.dot(X, W)  # f.shape = N, C
    f -= f.max(axis=1).reshape(N, 1)
    s = np.exp(f).sum(axis=1)
    loss = np.log(s).sum() - f[range(N), y].sum()

    counts = np.exp(f) / s.reshape(N, 1)
    counts[range(N), y] -= 1
    dW = np.dot(X.T, counts)
    loss = loss / N + 0.5 * reg * np.sum(W * W)
    dW = dW / N + reg * W
    return loss, dW