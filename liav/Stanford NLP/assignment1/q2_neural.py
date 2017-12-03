#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- N x Dx matrix, where each row is a training example.
    labels -- N x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension

    a1 = sig(data * W1 + b1)
    a2 = sig(a1 * W2 + b2)
    cost = -1 * sum(labels * log(a2))
    """

    # Unpack network parameters (do not modify)
    offset = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[offset:offset + Dx * H], (Dx, H))
    offset += Dx * H
    b1 = np.reshape(params[offset:offset + H], (1, H))
    offset += H
    W2 = np.reshape(params[offset:offset + H * Dy], (H, Dy))
    offset += H * Dy
    b2 = np.reshape(params[offset:offset + Dy], (1, Dy))

    # YOUR CODE HERE: forward propagation
    z1 = data.dot(W1) + b1  # (N x H)
    h = sigmoid(z1)  # (N x H)
    z2 = h.dot(W2) + b2
    yhat = softmax(z2)  # (N x Dy)
    cost = (-np.log(yhat) * labels).sum()

    # YOUR CODE HERE: backward propagation
    delta1 = yhat - labels  # N x Dy
    gradW2 = h.T.dot(delta1)  # H x Dy
    gradb2 = delta1.sum(axis=0, keepdims=True)
    assert gradW2.shape == W2.shape
    assert gradb2.size == b2.size, (gradb2.size,  b2.size)

    delta2 = delta1.dot(W2.T)  # N x H
    # originally did: sigmoid_grad(z1) here:
    delta3 = delta2 * sigmoid_grad(h)  # Wat?? element wise here?

    gradW1 = data.T.dot(delta3)
    gradb1 = delta3.sum(axis=0)
    assert gradW1.shape == W1.shape
    assert gradb1.size == b1.size, (gradb1.size,  b1.size)
    # END YOUR CODE
    # Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))
    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print('Running sanity check...')

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(
        lambda params: forward_backward_prop(data, labels, params, dimensions),
        params
    )


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print('Running your sanity checks...')
    # YOUR CODE HERE
    raise NotImplementedError
    # END YOUR CODE


if __name__ == '__main__':
    sanity_check()
    # your_sanity_checks()
