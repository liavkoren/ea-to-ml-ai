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
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation

    # sigmoid x_i input       ##, product first
    #sig_xi_prod = data.dot(W1)
    sig_xi_input = data.dot(W1) + b1
    ##print(data.shape)
    sig_xi_exp = np.exp(-sig_xi_input)
    sig_xi_den = 1 + sig_xi_exp
    sig_xi = 1 / sig_xi_den
    ##print(sig_xi.shape)

    # softmax numerator exponent input      ##, product first
    #sfmax_num_exp_prod = sig_xi.dot(W2)
    sfmax_num_exp_input = sig_xi.dot(W2) + b2
    # softmax numerator = softmax numerator exponent
    sfmax_num = np.exp(-sfmax_num_exp_input)
    ##print(sfmax_num.shape)
    # sigmoid x_j input       ##, product first
    #sig_xj_prod = data.dot(W1)
    sig_xj_input = data.dot(W1) + b1
    sig_xj_exp = np.exp(-sig_xj_input)
    sig_xj_den = 1 + sig_xj_exp
    sig_xj = 1 / sig_xj_den
    # softmax denominator exponent input      ##, product first
    #sfmax_den_exp_prod = sig_xj.dot(W2)
    sfmax_den_exp_input = sig_xj.dot(W2) + b2
    # softmax denominator exponent
    sfmax_den_exp = np.exp(-sfmax_den_exp_input)
    # softmax denominator = sum for all j's
    sfmax_den = np.sum(sfmax_den_exp, axis=0)
    sfmax_invden = 1 / sfmax_den        # invert!
    sfmax = np.multiply(sfmax_num, sfmax_invden)    # the softmax
    log_sfmax = np.log(sfmax)          # the log. numpy.log is ln.
    product = np.multiply(labels, log_sfmax) # multiply by y element-wise
    cost = -np.sum(product, axis=0)  # add up the rows

    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation                

    # TO DO: go over the chain rule stuff to make sure there are the right number of d[var]'s

    # sum and product out?    
    # sum: denominator squared and summed and negatived. Probably. (Check online)
    dlabels = log_sfmax
    dlog_sfmax = labels
    # numerator first
    dsfmax_num = sfmax_invden * dlog_sfmax
    dsfmax_num_exp = sfmax_num * dsfmax_num
    # "pocket" gradW2 and gradb2 portions from numerator
    gradW2 = -np.dot(sig_xi.T, dsfmax_num_exp)
    gradb2 = -1 * dsfmax_num_exp                #all grad b's might (not) need dimensionality adjustments
    # continuing:
    #dsfmax_den_exp_input = -numpy.multiply(sig_xi,(1-sig_xi)).dot(W2) * dsfmax_num_exp
    dsig_xi = -np.dot(dsfmax_num_exp, W2.T)
    dsig_xi_den = (-1) / (sig_xi_den**2) * dsig_xi
    #dsig_xi_exp = 1 / (sig_xi_input * sig_xi_exp) * dsig_xi_den    # first possible failure point
    dsig_xi_exp = sig_xi_input * sig_xi_exp * dsig_xi_den
    # "pocket" gradW1 and gradb1 portions from numerator
    gradW1 = np.dot(data.T, dsig_xi_exp)
    gradb1 = dsig_xi_exp
    # denominator
    dsfmax_invden = sfmax_num * dlog_sfmax
    dsfmax_den = (-1) / (sfmax_den**2) * dsfmax_invden  # similar possible failure point
    dsfmax_den_exp = sfmax_den_exp * sfmax_den_exp_input * dsfmax_num_exp      # in case needed: np.sum(sfmax_den_exp, axis=1)
    # "pocket" gradW2 and gradb2 portions from denomenator
    gradW2 += -np.dot(sig_xj.T, dsfmax_den_exp)
    gradb2 += -1 * dsfmax_den_exp
    # continuing:
    #dsfmax_den_exp_input = -numpy.multiply(sig_xi,(1-sig_xi)).dot(W2) * dsfmax_num_exp
    dsig_xj = -np.dot(dsfmax_num_exp, W2.T)
    dsig_xj_den = (-1) / (sig_xj_den**2) * dsig_xj
    dsig_xj_exp = sig_xj_input * sig_xj_exp * dsig_xj_den    # similar possible failure point
    gradW1 += np.dot(data.T, dsig_xj_exp)
    gradb1 += dsig_xj_exp

    # tryouts from doing the math, 2017.12.01
    sig_var = (np.dot(np.sum(data, axis=0), W1)) + b1   # 1 x 5
    sig_stuff = np.multiply((1 - sigmoid(sig_var)), sigmoid(sig_var))
    dW1_pt1 = np.multiply(W2.T, np.dot(np.sum(data, axis=0).T, sig_stuff))   # 10 x 5
    
    sig_pt2_var = np.dot(data, W1) + b1   # M x 5
    sig_pt2_plus = np.dot(sigmoid(sig_pt2_var), W2) + b2   # M x 10
    dW1_pt2_num = np.dot(np.multiply(np.exp(sig_pt2_plus), data).T, np.multiply(sigmoid(sig_pt2_var),(1-sigmoid(sig_pt2_var))) ) #10 x 5
    dW1_pt2_num = np.multiply(dW1_pt2_num, W2.T)   # 10 x 5
    dW1_pt2_den = np.sum(exp(sig_pt2_plus), axis=0) # 1 X 10
    dW1_pt_2_fraction = np.multiply(dW1_pt2_num, np.outer((1 / dW1_pt2_den).T, np.ones(5))) # 10 x 5

    gradW1 = np.multiply(np.outer(np.sum(labels, axis=0), np.ones(5), (dW1_pt1 - dW1_fraction)))

    db1_pt1 = np.multiply(W2.T, np.outer(np.ones(10), sig_stuff))
    db1_pt2_num = np.dot(np.exp(sig_pt2_plus).T, np.multiply(sigmoid(sig_pt2_var),(1-sigmoid(sig_pt2_var))) ) #10 x 5
    db1_pt2_num = np.multiply(db1_pt2_num, W2.T)   # 10 x 5
    db1_pt_2_fraction = np.multiply(db1_pt2_num, np.outer((1 / dW1_pt2_den).T, np.ones(5))) # 10 x 5
    gradb1 = np.multiply(np.outer(np.sum(labels, axis=0), np.ones(5), (db1_pt1 - db1_pt2_fraction)))

    dW2_pt1 = np.outer(sigmoid(sig_var).T, ones(10)) # 5 x 10
    dW2_pt2_num =  np.dot(sig_pt2_var.T, np.exp(sigmoid(sig_pt2_plus)))

    # D = 10, H = 5
    print(W2.shape)        #[5, 10]
    print(gradW2.shape)    #[5, 10]
    print(W1.shape)        #[10, 5]
    print(gradW1.shape)    #[10, 5]
    print(b2.shape)        #[1, 10]
    print(gradb2.shape)    #[20, 10] (!!!)
    print(b1.shape)        #[1, 5]
    print(gradb1.shape)    #[20, 5] (!!!)
    print('have printed shapes')
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck. 
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
