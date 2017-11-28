import numpy as np
from random import shuffle
from past.builtins import xrange

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
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather than first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  
  # loopy version, no adjustment of loss function calculation needed:
  # import pdb; pdb.set_trace()
  for i in xrange(num_train):
    for j in xrange(num_classes):
      if j == y[i]:
      	for k in xrange(num_classes):
          if k == j:
          	continue
          dW[:,j] -= X[i] * ((W[:,k]).T.dot(X[i]) - (W[:,y[i]].T.dot(X[i])) + 1 > 0) # T/F matrix for indicator function      	
      else:
        dW[:,j] += X[i] * (W[:,j].T.dot(X[i]) - (W[:,y[i]].T.dot(X[i])) + 1 > 0) # T/F matrix for indicator function      	
  dW = dW / num_train + reg * 2 * W 
  #import pdb; pdb.set_trace()
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.	

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  # one_hot = N X C



  outer1 = np.outer(np.ones(y.shape[0]), np.arange(W.shape[1]))
  outer2 = np.outer(y, np.ones(W.shape[1]))
  one_hot_correct_scores = (outer1 == outer2)

  # scores matrix = N X C
  scores = np.dot(X, W)
  correct_class_scores = np.dot(np.multiply(scores, one_hot_correct_scores), np.ones((W.shape[1], W.shape[1])))
  #correct_class_scores = np.multiply(np.diag(np.multiply(scores, one_hot_correct_scores)), ones(W.shape[1]).T)
  #ones = np.outer(ones())
  margin_matrix = np.maximum(0, scores - correct_class_scores + 1) - one_hot_correct_scores
 
  loss = np.sum(margin_matrix) / X.shape[0] + reg * np.sum(W * W)


  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  # dW = D X C
  # recall:      dW[:,j] += X[i]
  # recall:      dW[:,y[i]] -= X[i]
  # so:
  # dW += X * 


  # C X N
  # (W[:,j].T.dot(X[i]) - W[:,y[i]].T.dot(X[i]) + 1 > 0) * X[i]  <--- generalize this

  # scores matrix and the rest are N X C
  
  # zeroing out correct classes in entire W


  greater_than_zero_checks = ( np.maximum(0, scores - correct_class_scores + 1) - one_hot_correct_scores > 0 )

  #import pdb; pdb.set_trace()
  greater_than_zero_sums = np.sum(greater_than_zero_checks, axis=1).reshape(X.shape[0], 1)
  multiplier_matrix = np.ones(W.shape[1])
  dW -= np.dot(np.multiply(X, np.outer(greater_than_zero_sums, np.ones(W.shape[0]))).T, one_hot_correct_scores)

  dW += np.dot(X.T, greater_than_zero_checks)

  #dW -= np.dot(np.ones(X.shape[0], X.shape[0]), mar_mat_plus) - np.multiply(mar_mat_plus, one_hot_correct_scores)

  #greater_than_zero_checks = (scores - correct_class_scores + 1 - one_hot_correct_scores > 0)
  #dW -= np.dot(X.T, greater_than_zero_checks)

  #correct_class_subtractions =     
    #dW -= (greater_than_zero_checks and correct_class_checks) * X.T 
  #dW = np.dot(X.T, one_hot_correct_scores)

  dW = dW / X.shape[0] + reg * 2 * W
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################



  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

  #import pdb; pdb.set_trace()
