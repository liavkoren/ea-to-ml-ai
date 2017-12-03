import numpy as np
from random import shuffle
from past.builtins import xrange

def smax(scores, label):
  """ Takes a (1,C) score vector and the index of the correct score. """
  return np.exp(scores[label])/np.exp(scores).sum()


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
  N_train = X.shape[0]


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for data_index in xrange(N_train):
    scores = X[data_index].dot(W)  # (1, C)
    scores -= scores.max(keepdims=True)
    correct_class = y[data_index]
    loss += -np.log(smax(scores, correct_class))
    for class_index in xrange(num_classes):
      class_softmax = np.exp(scores[class_index])/sum(np.exp(scores))
      if class_index == correct_class:
        dW[:, class_index] += (class_softmax - 1) * X[data_index]
      else:
        dW[:, class_index] +=  class_softmax * X[data_index]

  loss /= N_train
  loss += reg * (W * W).sum()
  dW /= N_train
  dW += reg * 2 * W
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
  dW = np.zeros_like(W)  # (D, C)
  num_classes = W.shape[1]
  N_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)  # (N, C)
  scores -= scores.max(axis=1, keepdims=True)
  true_class_exp_scores = np.exp(scores[np.arange(N_train), y])
  exp_scores = np.exp(scores)
  sum_class_exp = np.sum(exp_scores, axis=1)
  loss = np.sum(-1 * np.log(true_class_exp_scores / sum_class_exp))

  # dW:
  softmax_denom = np.sum(exp_scores, axis=1, keepdims=True)
  softmax_scores = exp_scores / softmax_denom

  # First the incorrect class scores: Sum up all the scores for each class:
  # Scores is (N, C). We want to transform each (1, C) row of Scores into the
  # vector of class-softmaxes. This can then be dotted with X to the (D, C)
  # gradient matrix for the incorrect classes:

  incorrect_class_scores = np.copy(softmax_scores)
  # zero out the correct class scores:
  incorrect_class_scores[np.arange(N_train), y] = 0
  dW_incorrect = X.T.dot(incorrect_class_scores)

  # Now the correct class:
  correct_class_scores = np.copy(softmax_scores)
  correct_class_scores -= 1
  # zero out the incorrect class scores:
  zero_mask = np.ones_like(scores)
  zero_mask[np.arange(N_train), y] = False
  correct_class_scores[zero_mask.astype(bool)] = 0
  dW_correct  = X.T.dot(correct_class_scores)

  dW = dW_correct + dW_incorrect

  # Average + regularize:
  loss /= N_train
  loss += reg * (W * W).sum()
  dW /= N_train
  dW += reg * 2 * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

