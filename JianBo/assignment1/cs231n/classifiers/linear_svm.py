import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #计算j不等于yi的行的梯度
        dW[:, j] += X[i]
        #j=yi时的梯度
        dW[:, y[i]]+=(-X[i])

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train=X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  scores = X.dot(W)
  #这里得到一个500*10的矩阵,表示500个image的ground truth
  correct_class_score = scores[np.arange(num_train),y]
  #重复10次,得到500*10的矩阵,才可以和scores相加相减
  correct_class_score = np.reshape(np.repeat(correct_class_score,num_classes),(num_train,num_classes))
  margin = scores-correct_class_score+1.0
  margin[np.arange(num_train),y]=0

  loss = (np.sum(margin[margin > 0]))/num_train
  loss+=reg*np.sum(W*W)

  #gradient
  margin[margin>0]=1
  margin[margin<=0]=0

  row_sum = np.sum(margin, axis=1)                  # 1 by N
  margin[np.arange(num_train), y] = -row_sum
  dW += np.dot(X.T, margin)     # D by C
  # for xi in range(num_train):
  #   dW+=np.reshape(X[xi],(dW.shape[0],1))*\
  #       np.reshape(margin[xi],(1,dW.shape[1]))

  dW/=num_train
  dW += reg * W



  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
