#!/usr/bin/env python

import theano
import theano.tensor as T
import numpy as np

__author__ = "Sandeep Subramanian"
__maintainer__ = "Sandeep Subramanian"
__email__ = "sandeep.subramanian@gmail.com"

class Optimizer:

    """
    Optimization methods for backpropagation
    """

    def __init__(self):

        """
        __TODO__ add gradient clipping
        """

    def sgd(self, cost, params, lr=0.01):
        """
        Stochatic Gradient Descent.
        """
        lr = theano.shared(np.float64(lr).astype(theano.config.floatX))

        gradients = T.grad(cost, params)

        updates = []
        for param, gradient in zip(params, gradients):
            updates.append((param, param - lr * gradient))

        return updates

    def adagrad(self, cost, params, lr=0.01, epsilon=1e-6):
        """
        Adaptive Gradient Optimization.
        """
        lr = theano.shared(np.float64(lr).astype(theano.config.floatX))
        epsilon = theano.shared(np.float64(epsilon).astype(theano.config.floatX))

        gradients = T.grad(cost, params)

        updates = []
        for param, gradient in zip(params, gradients):
            accumulated_gradient = theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(np.float64), borrow=True)
            accumulated_gradient_new = accumulated_gradient + gradient  ** 2
            updates.append((accumulated_gradient, accumulated_gradient_new))
            updates.append((param, param - lr * gradient / T.sqrt(accumulated_gradient_new + epsilon)))
        return updates

    def rmsprop(self, cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
        """
        RMSProp - Root Mean Square
        Reference - http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        """

        lr = theano.shared(np.float64(lr).astype(theano.config.floatX))
        epsilon = theano.shared(np.float64(epsilon).astype(theano.config.floatX))
        rho = theano.shared(np.float64(rho).astype(theano.config.floatX))

        gradients = T.grad(cost, params)

        updates = []
        for param, gradient in zip(params, gradients):
            accumulated_gradient = theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(np.float64), borrow=True)
            accumulated_gradient_new = accumulated_gradient * rho + gradient ** 2 * (1 - rho)
            updates.append((accumulated_gradient, accumulated_gradient_new))
            updates.append((param, param - lr * gradient / T.sqrt(accumulated_gradient_new + epsilon)))
        return updates

    def adam(self, cost, params, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        """
        ADAM
        Reference - http://arxiv.org/pdf/1412.6980v8.pdf - Page 2
        """

        lr = theano.shared(np.float64(lr).astype(theano.config.floatX))
        epsilon = theano.shared(np.float64(epsilon).astype(theano.config.floatX))
        beta_1 = theano.shared(np.float64(beta_1).astype(theano.config.floatX))
        beta_2 = theano.shared(np.float64(beta_2).astype(theano.config.floatX))
        t = theano.shared(np.float64(1.0).astype(theano.config.floatX))

        gradients = T.grad(cost, params)

        updates = []
        for param, gradient in zip(params, gradients):
            param_value = param.get_value(borrow=True)
            m_tm_1 = theano.shared(np.zeros_like(param_value).astype(np.float64), borrow=True)
            v_tm_1 = theano.shared(np.zeros_like(param_value).astype(np.float64), borrow=True)

            m_t = beta_1 * m_tm_1 + (1 - beta_1) * gradient
            v_t = beta_2 * v_tm_1 + (1 - beta_2) * gradient ** 2

            m_hat = m_t / (1 - beta_1)
            v_hat = v_t / (1 - beta_2)

            updated_param = param - (lr * m_hat) / (T.sqrt(v_hat) + epsilon)
            updates.append((m_tm_1, m_t))
            updates.append((v_tm_1, v_t))
            updates.append((param, updated_param))

        updates.append((t, t + 1.0))
        return updates