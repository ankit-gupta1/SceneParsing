__author__ = 'ag3525'
import numpy
import theano

class LogisticRegression(object):
    def __init__(self, name, W, b, n_in, n_out):
        self.name = name

        if W is None:
            self.W = theano.shared(
                value=numpy.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        else:
            self.W = theano.shared(value=W, borrow=True)

        if b is None:
            self.b = theano.shared(
                value=numpy.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            self.b = theano.shared(value=b, borrow=True)

        self.p_y_given_x = None
        self.y_prediction = None
        self.params = [self.W, self.b]

    def compute_lr(self, input):
        self.p_y_given_x = theano.tensor.nnet.softmax(theano.tensor.dot(input, self.W) + self.b)
        self.y_prediction = theano.tensor.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        return -theano.tensor.mean(theano.tensor.log(self.p_y_given_x)[theano.tensor.arange(y.shape[0]), y])

    def errors(self, y):

        if y.ndim != self.y_prediction.ndim:
            raise TypeError(
                'y should have the same shape as self.y_prediction',
                ('y', y.type, 'y_prediction', self.y_prediction.type)
            )

        if y.dtype.startswith('int'):
            return theano.tensor.mean(theano.tensor.neq(self.y_prediction, y))
        else:
            raise NotImplementedError()