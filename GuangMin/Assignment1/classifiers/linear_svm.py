
import numpy as np

def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]
    loss /= num_train
    dW /= num_train

    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape)

    N = X.shape[0]
    scroes = X.dot(W)
    margin = scroes - scroes[range(0, N), y].reshape(-1, 1) + 1
    margin[range(N), y] = 0
    margin = (margin > 0) * margin
    loss += margin.sum() / N
    loss += 0.5 * reg * np.sum(W * W)

    counts = (margin > 0).astype('int')
    counts[range(N), y] -= np.sum(counts, axis=1)
    dW += np.dot(X.T, counts) / N + reg * W

    return loss, dW