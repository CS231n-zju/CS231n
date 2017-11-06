def svm_loss_naive1(W, X, y, reg):
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        dLi = np.zeros(W.shape)  # the gradient of L_i
        num_of_positive_margin = 0.0
        for k in range(num_classes):
            if k == y[i]:
                continue
            margin = scores[k] - correct_class_score + 1.0  # note delta = 1
            if margin > 0:
                loss += margin
                dLi[:, k] = X[i]
                num_of_positive_margin += 1.0
        # Set yi-th column (i.e. the remaining column) of dLi
        dLi[:, int(y[i])] = -num_of_positive_margin*X[i]
        dW += dLi
    # Add the gradient of the regularizer
    dW = (1.0/num_train)*dW + reg*W
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    return loss, dW
