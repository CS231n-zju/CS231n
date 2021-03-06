import numpy as np
from random import shuffle
# from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):

  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  flag = 0

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for i in range(num_train):
    correct_label = int(y[i])
    dw = np.zeros_like(W)
    scores = X[i].dot(W)
    correct = np.exp(scores[y[i]])
    exp_sum = np.sum(np.exp(scores))
    loss += -np.log(correct/exp_sum)

    for j in range(num_classes):
      if j != correct_label:
        dw[:,j] = X[i]*np.exp(scores[j])/exp_sum
    dw[:,correct_label] = X[i]*(correct/exp_sum - 1)
    dW += dw
    # print(dw)

  loss = loss/num_train + 0.5*reg*np.sum(W*W)
  # print(dW)
  dW = dW/num_train + reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)

  score_exp = np.exp(scores)

  exp_sum = np.sum(score_exp,axis = 1).reshape(num_train,1)

  correct_index = np.eye(num_classes)[y.reshape(len(y),)].astype('int')

  correct = np.exp(scores[np.where(correct_index)].reshape(num_train,1))

  loss = np.mean(-np.log(correct/exp_sum))

  dW = np.dot(X.T,score_exp/exp_sum-correct_index)

  dW = dW/num_train + reg*W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
