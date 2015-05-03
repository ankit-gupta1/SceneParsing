import os
import time
import math
import datetime
import sys

import numpy
import theano
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from PIL import Image
from scipy import misc

from logistic_regression import LogisticRegression
from hidden_layer import HiddenLayer
from load_data import generate_color_labels
from load_data import load_processed_img_data
from load_data import interpolate_tensor
from load_data import get_img_pyramids


class CNN_Layer(object):
    def __init__(self, name, W, b, filter_shape):
        self.name = name

        if W is None:
            rng = numpy.random.RandomState(23455)
            fan_in = numpy.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            self.W = theano.shared(value=W, borrow=True)

        if b is None:
            rng = numpy.random.RandomState(21281)
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = theano.shared(value=b, borrow=True)

        self.output = None
        self.params = [self.W, self.b]

    def compute_cnn(self, input, filter_shape, image_shape, border_mode, pool_size):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            border_mode=border_mode
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=pool_size,
            ignore_border=True,
        )

        self.output = theano.tensor.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]


def pad_zeros(ip, batch_size, dimension, height, width, filter_size):
    padding = int(math.floor(filter_size / 2))
    z = theano.tensor.zeros((batch_size, dimension, height + filter_size - 1, width + filter_size - 1),
                            dtype=theano.config.floatX)
    z = theano.tensor.set_subtensor(z[:, :, padding:(height + padding), padding:(width + padding)], ip)
    return z


def forward_propagation(layer0,
                        layer1,
                        layer2,
                        layer3,
                        layer4,
                        x_by_1,
                        x_by_2,
                        x_by_4,
                        num_kernels,
                        batch_size,
                        filter_size,
                        is_multi_scale,
                        height,
                        width,
                        use_interpolation,
                        use_hidden_layer):
    layer0_input = x_by_1.reshape((batch_size, 3, height, width))
    layer0_input = pad_zeros(layer0_input, batch_size, 3, height, width, filter_size)
    layer0.compute_cnn(
        input=layer0_input,
        filter_shape=(num_kernels[0], 3, filter_size, filter_size),
        image_shape=(batch_size, 3, height + filter_size - 1, width + filter_size - 1),
        border_mode='valid',
        pool_size=(2, 2)
    )

    layer1_input = pad_zeros(layer0.output, batch_size, num_kernels[0], height / 2, width / 2, filter_size)
    layer1.compute_cnn(
        input=layer1_input,
        filter_shape=(num_kernels[1], num_kernels[0], filter_size, filter_size),
        image_shape=(batch_size, num_kernels[0], (height / 2) + filter_size - 1, (width / 2) + filter_size - 1),
        border_mode='valid',
        pool_size=(2, 2),
    )

    layer2_input = pad_zeros(layer1.output, batch_size, num_kernels[1], height / 4, width / 4, filter_size)
    layer2.compute_cnn(
        input=layer2_input,
        filter_shape=(num_kernels[2], num_kernels[1], filter_size, filter_size),
        image_shape=(batch_size, num_kernels[1], (height / 4) + filter_size - 1, (width / 4) + filter_size - 1),
        border_mode='valid',
        pool_size=(1, 1)
    )

    if use_interpolation is True:
        layer2.output = interpolate_tensor(layer2.output, 4, axis=3)
        layer2.output = interpolate_tensor(layer2.output, 4, axis=2)
    else:
        layer2.output = theano.tensor.repeat(layer2.output, 4, axis=2)
        layer2.output = theano.tensor.repeat(layer2.output, 4, axis=3)

    layer2.output = layer2.output.dimshuffle(0, 2, 3, 1)
    mlp_input = layer2.output.reshape((batch_size * height * width, num_kernels[2]))

    if is_multi_scale is True:
        layer0_input = x_by_2.reshape((batch_size, 3, height / 2, width / 2))
        layer0_input = pad_zeros(layer0_input, batch_size, 3, height / 2, width / 2, filter_size)
        layer0.compute_cnn(
            input=layer0_input,
            filter_shape=(num_kernels[0], 3, filter_size, filter_size),
            image_shape=(batch_size, 3, (height / 2) + filter_size - 1, (width / 2) + filter_size - 1),
            border_mode='valid',
            pool_size=(2, 2)
        )

        layer1_input = pad_zeros(layer0.output, batch_size, num_kernels[0], height / 4, width / 4, filter_size)
        layer1.compute_cnn(
            input=layer1_input,
            filter_shape=(num_kernels[1], num_kernels[0], filter_size, filter_size),
            image_shape=(batch_size, num_kernels[0], (height / 4) + filter_size - 1, (width / 4) + filter_size - 1),
            border_mode='valid',
            pool_size=(2, 2)
        )

        layer2_input = pad_zeros(layer1.output, batch_size, num_kernels[1], (height / 8), (width / 8), filter_size)
        layer2.compute_cnn(
            input=layer2_input,
            filter_shape=(num_kernels[2], num_kernels[1], filter_size, filter_size),
            image_shape=(batch_size, num_kernels[1], (height / 8) + filter_size - 1, (width / 8) + filter_size - 1),
            border_mode='valid',
            pool_size=(1, 1)
        )

        if use_interpolation is True:
            layer2.output = interpolate_tensor(layer2.output, 8, axis=3)
            layer2.output = interpolate_tensor(layer2.output, 8, axis=2)
        else:
            layer2.output = theano.tensor.repeat(layer2.output, 8, axis=2)
            layer2.output = theano.tensor.repeat(layer2.output, 8, axis=3)

        layer2.output = layer2.output.dimshuffle(0, 2, 3, 1)
        mlp_input1 = layer2.output.reshape((batch_size * height * width, num_kernels[2]))

        layer0_input = x_by_4.reshape((batch_size, 3, height / 4, width / 4))
        layer0_input = pad_zeros(layer0_input, batch_size, 3, height / 4, width / 4, filter_size)
        layer0.compute_cnn(
            input=layer0_input,
            filter_shape=(num_kernels[0], 3, filter_size, filter_size),
            image_shape=(batch_size, 3, (height / 4) + filter_size - 1, (width / 4) + filter_size - 1),
            border_mode='valid',
            pool_size=(2, 2)
        )

        layer1_input = pad_zeros(layer0.output, batch_size, num_kernels[0], height / 8, width / 8, filter_size)
        layer1.compute_cnn(
            input=layer1_input,
            filter_shape=(num_kernels[1], num_kernels[0], filter_size, filter_size),
            image_shape=(batch_size, num_kernels[0], (height / 8) + filter_size - 1, (width / 8) + filter_size - 1),
            border_mode='valid',
            pool_size=(2, 2)
        )

        layer2_input = pad_zeros(layer1.output, batch_size, num_kernels[1], (height / 16), (width / 16), filter_size)
        layer2.compute_cnn(
            input=layer2_input,
            filter_shape=(num_kernels[2], num_kernels[1], filter_size, filter_size),
            image_shape=(batch_size, num_kernels[1], (height / 16) + filter_size - 1, (width / 16) + filter_size - 1),
            border_mode='valid',
            pool_size=(1, 1)
        )

        if use_interpolation is True:
            layer2.output = interpolate_tensor(layer2.output, 16, axis=3)
            layer2.output = interpolate_tensor(layer2.output, 16, axis=2)
        else:
            layer2.output = theano.tensor.repeat(layer2.output, 16, axis=2)
            layer2.output = theano.tensor.repeat(layer2.output, 16, axis=3)

        layer2.output = layer2.output.dimshuffle(0, 2, 3, 1)
        mlp_input2 = layer2.output.reshape((batch_size * height * width, num_kernels[2]))
        mlp_input = theano.tensor.concatenate((mlp_input, mlp_input1, mlp_input2), axis=1)

    if use_hidden_layer is True:
        layer3.compute_hl(input=mlp_input, activation=theano.tensor.tanh)
        layer4.compute_lr(
            input=layer3.output
        )
    else:
        layer4.compute_lr(
            input=mlp_input
        )


def train_CNN_mini_batch(learning_rate,
                         n_epochs,
                         num_kernels,
                         batch_size,
                         filter_size,
                         is_multi_scale,
                         num_of_classes,
                         height,
                         width,
                         use_interpolation,
                         use_hidden_layer):
    train_set_x_by_1, train_set_y, valid_set_x_by_1, valid_set_y, test_set_x_by_1, test_set_y, train_set_x_by_2, \
    train_set_x_by_4, valid_set_x_by_2, valid_set_x_by_4, test_set_x_by_2, test_set_x_by_4 \
        = load_processed_img_data()

    n_train_batches = train_set_x_by_1.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x_by_1.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x_by_1.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    index = theano.tensor.lscalar()
    x_by_1 = theano.tensor.ftensor4('x_by_1')
    x_by_2 = theano.tensor.ftensor4('x_by_2')
    x_by_4 = theano.tensor.ftensor4('x_by_4')

    y = theano.tensor.ivector('y')

    print '... initialize the model'

    cnn_dir = 'models/CNN_'
    if is_multi_scale is True:
        cnn_dir += 'M_'
    else:
        cnn_dir += 'S_'

    if use_hidden_layer is True:
        cnn_dir += 'H_'
    else:
        cnn_dir += 'L_'

    if use_interpolation is True:
        cnn_dir += 'I_'
    else:
        cnn_dir += 'N_'

    cnn_dir = cnn_dir + str(num_kernels[0]) + '_' + str(num_kernels[1]) + '_' + str(num_kernels[2]) + '_' + str(
        batch_size) + '_'
    curr_date = str(datetime.date.today())
    curr_date = curr_date.replace('-', '_')
    cnn_dir = cnn_dir + curr_date + str(time.strftime('_%H_%M_%S'))

    print 'CNN model is ', cnn_dir

    if not os.path.exists(cnn_dir):
        os.makedirs(cnn_dir)

    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(cnn_dir + '/log.txt', 'w')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

    sys.stdout = Logger()

    layer0 = CNN_Layer(
        name='Layer_0',
        W=None,
        b=None,
        filter_shape=(num_kernels[0], 3, filter_size, filter_size),
    )

    layer1 = CNN_Layer(
        name='Layer_1',
        W=None,
        b=None,
        filter_shape=(num_kernels[1], num_kernels[0], filter_size, filter_size),
    )

    layer2 = CNN_Layer(
        name='Layer_2',
        W=None,
        b=None,
        filter_shape=(num_kernels[2], num_kernels[1], filter_size, filter_size),
    )

    layer3 = HiddenLayer(
        name='Layer_3',
        W=None,
        b=None,
        n_in=num_kernels[2] * 3 if is_multi_scale is True else num_kernels[2],
        n_out=num_kernels[2] * 4 if is_multi_scale is True else num_kernels[2] * 2,
        activation=theano.tensor.tanh
    )

    if is_multi_scale and use_hidden_layer:
        layer4_in = num_kernels[2] * 4
    elif is_multi_scale and not use_hidden_layer:
        layer4_in = num_kernels[2] * 3
    elif not is_multi_scale and use_hidden_layer:
        layer4_in = num_kernels[2] * 2
    else:
        layer4_in = num_kernels[2]

    layer4 = LogisticRegression(
        name='Layer_4',
        W=None,
        b=None,
        n_in=layer4_in,
        n_out=num_of_classes,
    )

    forward_propagation(
        layer0=layer0,
        layer1=layer1,
        layer2=layer2,
        layer3=layer3,
        layer4=layer4,
        x_by_1=x_by_1,
        x_by_2=x_by_2,
        x_by_4=x_by_4,
        num_kernels=num_kernels,
        batch_size=batch_size,
        filter_size=filter_size,
        is_multi_scale=is_multi_scale,
        height=height,
        width=width,
        use_interpolation=use_interpolation,
        use_hidden_layer=use_hidden_layer
    )

    if use_hidden_layer is True:
        L2_norm = (layer4.W ** 2).sum() + (layer3.W ** 2).sum() + (layer2.W ** 2).sum() + (layer1.W ** 2).sum() + (
        layer0.W ** 2).sum()
    else:
        L2_norm = (layer4.W ** 2).sum() + (layer2.W ** 2).sum() + (layer1.W ** 2).sum() + (layer0.W ** 2).sum()

    regularization = 0.00001
    cost = layer4.negative_log_likelihood(y) + (regularization * L2_norm)

    if is_multi_scale is True:
        test_model = theano.function(
            [index],
            layer4.errors(y),
            givens={
                x_by_1: test_set_x_by_1[index * batch_size: (index + 1) * batch_size],
                x_by_2: test_set_x_by_2[index * batch_size: (index + 1) * batch_size],
                x_by_4: test_set_x_by_4[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size * height * width: (index + 1) * batch_size * height * width]
            }
        )
    else:
        test_model = theano.function(
            [index],
            layer4.errors(y),
            givens={
                x_by_1: test_set_x_by_1[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size * height * width: (index + 1) * batch_size * height * width]
            }
        )

    if is_multi_scale is True:
        validate_model = theano.function(
            [index],
            layer4.errors(y),
            givens={
                x_by_1: valid_set_x_by_1[index * batch_size: (index + 1) * batch_size],
                x_by_2: valid_set_x_by_2[index * batch_size: (index + 1) * batch_size],
                x_by_4: valid_set_x_by_4[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size * height * width: (index + 1) * batch_size * height * width]
            }
        )
    else:
        validate_model = theano.function(
            [index],
            layer4.errors(y),
            givens={
                x_by_1: valid_set_x_by_1[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size * height * width: (index + 1) * batch_size * height * width]
            }
        )

    if use_hidden_layer is True:
        params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
    else:
        params = layer4.params + layer2.params + layer1.params + layer0.params

    grads = theano.tensor.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    if is_multi_scale is True:
        train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x_by_1: train_set_x_by_1[index * batch_size: (index + 1) * batch_size],
                x_by_2: train_set_x_by_2[index * batch_size: (index + 1) * batch_size],
                x_by_4: train_set_x_by_4[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size * width * height: (index + 1) * batch_size * width * height]
            }
        )
    else:
        train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x_by_1: train_set_x_by_1[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size * width * height: (index + 1) * batch_size * width * height]
            }
        )

    print '... training the model'
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    best_layer_0_W = numpy.zeros_like(layer0.W.get_value())
    best_layer_0_b = numpy.zeros_like(layer0.b.get_value())
    best_layer_1_W = numpy.zeros_like(layer1.W.get_value())
    best_layer_1_b = numpy.zeros_like(layer1.b.get_value())
    best_layer_2_W = numpy.zeros_like(layer2.W.get_value())
    best_layer_2_b = numpy.zeros_like(layer2.b.get_value())
    best_layer_3_W = numpy.zeros_like(layer3.W.get_value())
    best_layer_3_b = numpy.zeros_like(layer3.b.get_value())
    best_layer_4_W = numpy.zeros_like(layer4.W.get_value())
    best_layer_4_b = numpy.zeros_like(layer4.b.get_value())

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for mini_batch_index in xrange(n_train_batches):

            start = time.clock()
            iter = (epoch - 1) * n_train_batches + mini_batch_index
            cost_ij = train_model(mini_batch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, mini-batch %i/%i, validation error %f %%' %
                      (epoch, mini_batch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # save best filters
                    best_layer_0_W = layer0.W.get_value()
                    best_layer_0_b = layer0.b.get_value()
                    best_layer_1_W = layer1.W.get_value()
                    best_layer_1_b = layer1.b.get_value()
                    best_layer_2_W = layer2.W.get_value()
                    best_layer_2_b = layer2.b.get_value()
                    best_layer_3_W = layer3.W.get_value()
                    best_layer_3_b = layer3.b.get_value()
                    best_layer_4_W = layer4.W.get_value()
                    best_layer_4_b = layer4.b.get_value()

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]

                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, mini-batch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, mini_batch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

            print 'training @ iter = %d, time taken = %f' % (iter, (time.clock() - start))

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    if not os.path.exists(cnn_dir + '/params'):
        os.makedirs(cnn_dir + '/params')

    numpy.save(cnn_dir + '/params/layer_0_W.npy', best_layer_0_W)
    numpy.save(cnn_dir + '/params/layer_0_b.npy', best_layer_0_b)
    numpy.save(cnn_dir + '/params/layer_1_W.npy', best_layer_1_W)
    numpy.save(cnn_dir + '/params/layer_1_b.npy', best_layer_1_b)
    numpy.save(cnn_dir + '/params/layer_2_W.npy', best_layer_2_W)
    numpy.save(cnn_dir + '/params/layer_2_b.npy', best_layer_2_b)
    numpy.save(cnn_dir + '/params/layer_3_W.npy', best_layer_3_W)
    numpy.save(cnn_dir + '/params/layer_3_b.npy', best_layer_3_b)
    numpy.save(cnn_dir + '/params/layer_4_W.npy', best_layer_4_W)
    numpy.save(cnn_dir + '/params/layer_4_b.npy', best_layer_4_b)
    numpy.save(cnn_dir + '/params/filer_kernels.npy', num_kernels)
    numpy.save(cnn_dir + '/params/filter_size.npy', filter_size)

    return cnn_dir


def generate_segmented_image_tensors(img_by_1,
                                     img_by_2,
                                     img_by_4,
                                     model_dir,
                                     batch_size,
                                     height,
                                     width,
                                     num_of_classes):
    layer_0_W = numpy.load(model_dir + '/params/layer_0_W.npy')
    layer_0_b = numpy.load(model_dir + '/params/layer_0_b.npy')
    layer_1_W = numpy.load(model_dir + '/params/layer_1_W.npy')
    layer_1_b = numpy.load(model_dir + '/params/layer_1_b.npy')
    layer_2_W = numpy.load(model_dir + '/params/layer_2_W.npy')
    layer_2_b = numpy.load(model_dir + '/params/layer_2_b.npy')
    layer_3_W = numpy.load(model_dir + '/params/layer_3_W.npy')
    layer_3_b = numpy.load(model_dir + '/params/layer_3_b.npy')
    layer_4_W = numpy.load(model_dir + '/params/layer_4_W.npy')
    layer_4_b = numpy.load(model_dir + '/params/layer_4_b.npy')
    num_kernels = numpy.load(model_dir + '/params/filer_kernels.npy')
    filter_size = numpy.load(model_dir + '/params/filter_size.npy')

    if model_dir[11] == 'M':
        is_multi_scale = True
    elif model_dir[11] == 'S':
        is_multi_scale = False
    else:
        return NotImplemented

    if model_dir[13] == 'H':
        use_hidden_layer = True
    elif model_dir[13] == 'L':
        use_hidden_layer = False
    else:
        return NotImplemented

    if model_dir[15] == 'I':
        use_interpolation = True
    elif model_dir[13] == 'L':
        use_interpolation = False
    else:
        return NotImplemented

    layer0 = CNN_Layer(
        name='Layer_0',
        W=layer_0_W,
        b=layer_0_b,
        filter_shape=(num_kernels[0], 3, filter_size, filter_size),
    )

    layer1 = CNN_Layer(
        name='Layer_1',
        W=layer_1_W,
        b=layer_1_b,
        filter_shape=(num_kernels[1], num_kernels[0], filter_size, filter_size),
    )

    layer2 = CNN_Layer(
        name='Layer_2',
        W=layer_2_W,
        b=layer_2_b,
        filter_shape=(num_kernels[2], num_kernels[1], filter_size, filter_size),
    )

    layer3 = HiddenLayer(
        name='Layer_3',
        W=layer_3_W,
        b=layer_3_b,
        n_in=num_kernels[2] * 3 if is_multi_scale is True else num_kernels[2],
        n_out=num_kernels[2] * 4 if is_multi_scale is True else num_kernels[2] * 2,
        activation=theano.tensor.tanh
    )

    layer4 = LogisticRegression(
        name='Layer_4',
        W=layer_4_W,
        b=layer_4_b,
        n_in=num_kernels[2] * 4 if is_multi_scale is True else num_kernels[2] * 2,
        n_out=num_of_classes,
    )

    x_by_1 = theano.tensor.ftensor4('x_by_1')
    x_by_2 = theano.tensor.ftensor4('x_by_2')
    x_by_4 = theano.tensor.ftensor4('x_by_4')

    forward_propagation(
        layer0=layer0,
        layer1=layer1,
        layer2=layer2,
        layer3=layer3,
        layer4=layer4,
        x_by_1=x_by_1,
        x_by_2=x_by_2,
        x_by_4=x_by_4,
        num_kernels=num_kernels,
        batch_size=batch_size,
        filter_size=filter_size,
        is_multi_scale=is_multi_scale,
        height=height,
        width=width,
        use_interpolation=use_interpolation,
        use_hidden_layer=use_hidden_layer,
    )

    # create a function to compute the mistakes that are made by the model
    if is_multi_scale is True:
        test_model = theano.function(
            [x_by_1, x_by_2, x_by_4], layer4.y_prediction
        )
    else:
        test_model = theano.function(
            [x_by_1], layer4.y_prediction
        )

    if is_multi_scale is True:
        op = test_model(img_by_1, img_by_2, img_by_4)
    else:
        op = test_model(img_by_1)

    y = theano.tensor.reshape(op, (batch_size, height, width))
    return y.eval()


def generate_segmented_images_multiscale(model_dir, num, height, width):
    counter = 0
    L = []

    X_by_1 = numpy.zeros((num, 3, height, width), dtype=theano.config.floatX)
    X_by_2 = numpy.zeros((num, 3, height / 2, width / 2), dtype=theano.config.floatX)
    X_by_4 = numpy.zeros((num, 3, height / 4, width / 4), dtype=theano.config.floatX)

    if model_dir[11] == 'M':
        is_multi_scale = True
    elif model_dir[11] == 'S':
        is_multi_scale = False
    else:
        return NotImplemented

    for file in os.listdir('dataset/images_formatted/'):
        if file.endswith('.jpg'):

            L.append(file[0: file.find('.')])
            temp_by_1, temp_by_2, temp_by_4 = get_img_pyramids(misc.imread('dataset/images_formatted/' + file))

            print 'Processed image number ', counter

            X_by_1[counter, :, :, :] = temp_by_1
            X_by_2[counter, :, :, :] = temp_by_2
            X_by_4[counter, :, :, :] = temp_by_4

            counter += 1
            if counter >= num:
                break

    label_data = generate_segmented_image_tensors(
        img_by_1=X_by_1,
        img_by_2=X_by_2,
        img_by_4=X_by_4,
        model_dir=model_dir,
        batch_size=num,
        height=height,
        width=width,
        num_of_classes=9,
    )

    print label_data.shape

    if is_multi_scale is True:
        gray_scale_dir = model_dir + '/segmented_multi_scale_gray/'
        color_scale_dir = model_dir + '/segmented_multi_scale_colored/'

    else:
        gray_scale_dir = model_dir + '/segmented_single_scale_gray/'
        color_scale_dir = model_dir + '/segmented_single_scale_colored/'

    if not os.path.exists(gray_scale_dir):
        os.makedirs(gray_scale_dir)

    if not os.path.exists(color_scale_dir):
        os.makedirs(color_scale_dir)

    for i in range(0, counter):
        temp = label_data[i, :, :]
        temp_gray = (temp * 20) + 20
        file_name = gray_scale_dir + str(L[i]) + '_label.png'
        print 'Generating segmented image ', file_name
        misc.imsave(file_name, temp_gray)

        colored = generate_color_labels(temp)
        misc.imsave('models/temp_colored.jpg', colored)
        file_name = 'dataset/images_formatted/' + str(L[i]) + '.jpg'
        image1 = Image.open(file_name)
        image2 = Image.open('models/temp_colored.jpg')
        blend = Image.blend(image1, image2, 0.3)
        file_name = color_scale_dir + str(L[i]) + '_label.png'
        blend.save(file_name)


def generate_segmented_images_original(model_dir):
    gray_scale_dir = model_dir + '/grayscale_original/'
    color_scale_dir = model_dir + '/colored_original/'

    if not os.path.exists(gray_scale_dir):
        os.makedirs(gray_scale_dir)

    if not os.path.exists(color_scale_dir):
        os.makedirs(color_scale_dir)

    for file in os.listdir('dataset/labels_formatted/'):
        if file.endswith('.txt'):

            fileNumber = file[0: file.find('.')]
            tempLabels = numpy.zeros((240 * 320), numpy.int32)
            labelPos = 0

            with open('dataset/labels_formatted/' + fileNumber + '.regions.txt', 'r') as labelFile:
                while True:
                    c = labelFile.read(1)
                    if not c:
                        break

                    if c == '-':
                        c = labelFile.read(1)
                        tempLabels[labelPos] = 0
                        labelPos += 1

                    elif c != ' ' and c != '\n':
                        tempLabels[labelPos] = int(c) + 1
                        labelPos += 1

            tempLabels = numpy.reshape(tempLabels, (240, 320))
            tempLabels1 = (tempLabels * 20) + 20
            file_name = gray_scale_dir + str(fileNumber) + '_label.png'
            print 'Generating original segmented image ', file_name
            misc.imsave(file_name, tempLabels1)
            colored = generate_color_labels(tempLabels)
            misc.imsave('models/temp_colored.jpg', colored)
            file_name = 'dataset/images_formatted/' + str(fileNumber) + '.jpg'
            image1 = Image.open(file_name)
            image2 = Image.open('models/temp_colored.jpg')
            blend = Image.blend(image1, image2, 0.3)
            file_name = color_scale_dir + str(fileNumber) + '_label.png'
            blend.save(file_name)


if __name__ == '__main__':
    model_dir = train_CNN_mini_batch(learning_rate=0.1,
                                     n_epochs=50,
                                     num_kernels=[16, 64, 128],
                                     batch_size=5,
                                     filter_size=7,
                                     is_multi_scale=False,
                                     num_of_classes=9,
                                     height=240,
                                     width=320,
                                     use_interpolation=True,
                                     use_hidden_layer=False)

    # model_dir = 'models/CNN_S_L_N_16_64_128_5_2015_05_03_09_26_58'
    generate_segmented_images_original(model_dir=model_dir)
    generate_segmented_images_multiscale(model_dir=model_dir,
                                         num=100,
                                         height=240,
                                         width=320)