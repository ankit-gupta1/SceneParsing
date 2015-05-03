__author__ = 'ag3525'

import theano
import numpy

class HiddenLayer(object):
    def __init__(self, name, W, b, n_in, n_out, activation=theano.tensor.tanh):
        self.name = name
        if W is None:
            rng = numpy.random.RandomState(48810)
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            self.W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            self.W = theano.shared(value=W, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            self.b = theano.shared(value=b, name='b', borrow=True)

        self.output = None
        self.params = [self.W, self.b]

    def compute_hl(self, input, activation=theano.tensor.tanh):
        lin_output = theano.tensor.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )