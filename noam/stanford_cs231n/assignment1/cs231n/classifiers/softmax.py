import numpy as np
from random import shuffle
from past.builtins import xrange
#r
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

  #import pdb; pdb.set_trace()
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in xrange(num_train):
    # compute vector of scores
    f_i = X[i].dot(W)

    # normalization trick
    f_i -= np.max(f_i)

    # compute or add to loss (to be divided later)
    sum_j = np.sum(np.exp(f_i))
    sfmax = lambda k: np.exp(f_i[k]) / sum_j
    loss += -np.log(sfmax([y[i]]))

    # Compute gradient
    # Note subtraction of correct-class cases
    for k in xrange(W.shape[1]):
      sfmax_k = sfmax(k)
      dW[:, k] += (sfmax_k - (k == y[i])) * X[i]

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += reg * 2 * W

  return loss, dW
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################



def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]
  f = X.dot(W)
  f -= np.max(f, axis=1, keepdims=True)
  sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
  sfmax = np.exp(f)/sum_f
  loss = np.sum(-np.log(sfmax[np.arange(num_train), y]))

  # indicator function
  ind_func = np.zeros_like(sfmax)
  ind_func[np.arange(num_train), y] = 1

  dW = X.T.dot(sfmax - ind_func)

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += reg * 2 * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

