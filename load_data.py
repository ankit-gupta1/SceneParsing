__author__ = 'ag3525'
import numpy
import os
import theano
import cv2
import scipy

from PIL import Image
from scipy import misc

def format_img():
    for file in os.listdir('dataset/images/'):
        if file.endswith('.jpg'):
            print(file)
            img_mod = misc.imread('dataset/images/' + file)

            height = img_mod.shape[0]
            width = img_mod.shape[1]
            y = numpy.zeros((height, width, 3), 'uint8')

            if img_mod.shape[0] >= height and img_mod.shape[1] >= width:
                y = img_mod[0:height, 0:width, :]

            elif img_mod.shape[0] >= height and img_mod.shape[1] < width:
                y[:, 0:img_mod.shape[1], :] = img_mod[0:height, 0:img_mod.shape[1], :]

            elif img_mod.shape[0] < height and img_mod.shape[1] >= width:
                y[0:img_mod.shape[0], :, :] = img_mod[0:img_mod.shape[0], 0:width, :]

            elif img_mod.shape[0] < height and img_mod.shape[1] < width:
                y[0:img_mod.shape[0], 0:img_mod.shape[1], :] = img_mod

            img_array = Image.fromarray(y)
            img_array.save('dataset/images_formatted/' + file)

def format_labels(height, width):
    NoError = True
    for file in os.listdir('dataset/labels/'):
        if file.endswith('.txt'):
            chars = words = lines = 0
            print file

            with open('dataset/labels/' + file, 'r') as in_file:
                for line in in_file:
                    lines += 1
                    words += len(line.split())
                    chars += len(line)

            numRows = lines
            numCols = words / lines

            opFileName = 'dataset/labels_formatted/' + file
            opObj = open(opFileName, 'w')

            with open('dataset/labels/' + file, 'r') as f:
                rowsWrite = 0
                colWrite = 0
                doRead = 1
                itsMinus = 0

                while True:
                    if doRead == 1:
                        c = f.read(1)
                        if c == '-':
                            c = f.read(1)
                            itsMinus = 1

                    if not c:
                        break

                    if c != ' ' and colWrite < width:
                        if colWrite < numCols and colWrite < width and rowsWrite < height:
                            if itsMinus == 0 and c != '\n':
                                opObj.write(c)
                                opObj.write(' ')
                            elif itsMinus == 1:
                                opObj.write('-1')
                                opObj.write(' ')
                                itsMinus = 0

                        elif colWrite < width and colWrite >= numCols and rowsWrite < height:
                            doRead = 0
                            opObj.write('-1')
                            opObj.write(' ')

                        colWrite += 1

                    if colWrite >= width:
                        if rowsWrite < height:
                            opObj.write('\n')

                        rowsWrite += 1
                        colWrite = 0
                        doRead = 1
                        itsMinus = 0

                    if rowsWrite >= numRows - 1:
                        break

                if rowsWrite < height:
                    for i in range(rowsWrite, height):
                        for j in range(0, width):
                            opObj.write('-1')
                            opObj.write(' ')
                        opObj.write('\n')

            opObj.close()

    for file in os.listdir('dataset/labels_formatted/'):
        if file.endswith('.txt'):
            chars = words = lines = 0
            print file

            maxL = 0
            with open('dataset/labels_formatted/' + file, 'r') as in_file:
                for line in in_file:
                    lines += 1
                    words += len(line.split())
                    if len(line.split()) > maxL:
                        maxL = len(line.split())

                    chars += len(line)

            numRows = lines
            numCols = maxL

            print numRows, numCols
            if numCols != width or numRows != height:
                NoError = False

    return NoError

def normalize(img, block_size):
    height = img.shape[0]
    width = img.shape[1]

    temp = numpy.zeros((height, width, 3), dtype=theano.config.floatX)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            y = img[i:(i + block_size), j:(j + block_size), 0]
            u = img[i:(i + block_size), j:(j + block_size), 1]
            v = img[i:(i + block_size), j:(j + block_size), 2]

            y = (y - y.mean()) / (1 if y.var() == 0 else y.var()**(0.5))
            u = (u - u.mean()) / (1 if u.var() == 0 else u.var()**(0.5))
            v = (v - v.mean()) / (1 if v.var() == 0 else v.var()**(0.5))

            temp[i:(i + block_size), j:(j + block_size), 0] = y
            temp[i:(i + block_size), j:(j + block_size), 1] = u
            temp[i:(i + block_size), j:(j + block_size), 2] = v

    return temp

def contrast_normalize_tensors(img):
    img_y = img[:, :, 0]
    img_d = numpy.zeros_like(img_y, dtype=numpy.uint8)
    cv2.equalizeHist(src=img_y, dst=img_d)
    img[:, :, 0] = img_d
    return img

def get_img_pyramids(img):
    height = img.shape[0]
    width = img.shape[1]

    temp_by_1 = numpy.zeros((3, height, width), dtype=theano.config.floatX)
    temp_by_2 = numpy.zeros((3, height / 2, width / 2), dtype=theano.config.floatX)
    temp_by_4 = numpy.zeros((3, height / 4, width / 4), dtype=theano.config.floatX)

    img_yuv_by_1 = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv_by_2 = cv2.pyrDown(src=img_yuv_by_1)
    img_yuv_by_4 = cv2.pyrDown(src=img_yuv_by_2)

    img_yuv_by_1 = contrast_normalize_tensors(img_yuv_by_1)
    img_yuv_by_2 = contrast_normalize_tensors(img_yuv_by_2)
    img_yuv_by_4 = contrast_normalize_tensors(img_yuv_by_4)

    img_yuv_by_1.astype(dtype=theano.config.floatX)
    img_yuv_by_2.astype(dtype=theano.config.floatX)
    img_yuv_by_4.astype(dtype=theano.config.floatX)

    img_by_1 = normalize(img_yuv_by_1, 16)
    temp_by_1[0, :, :] = img_by_1[:, :, 0]
    temp_by_1[1, :, :] = img_by_1[:, :, 1]
    temp_by_1[2, :, :] = img_by_1[:, :, 2]

    img_by_2 = normalize(img_yuv_by_2, 16)
    temp_by_2[0, :, :] = img_by_2[:, :, 0]
    temp_by_2[1, :, :] = img_by_2[:, :, 1]
    temp_by_2[2, :, :] = img_by_2[:, :, 2]

    img_by_4 = normalize(img_yuv_by_4, 16)
    temp_by_4[0, :, :] = img_by_4[:, :, 0]
    temp_by_4[1, :, :] = img_by_4[:, :, 1]
    temp_by_4[2, :, :] = img_by_4[:, :, 2]

    return temp_by_1, temp_by_2, temp_by_4

def pre_process_img_and_labels_multi_scale(height, width, num_train_set, num_validation_set, num_test_set):
    counter = 0

    trainX_by_1 = numpy.zeros((num_train_set, 3, height, width), dtype=theano.config.floatX)
    valX_by_1 = numpy.zeros((num_validation_set, 3, height, width), dtype=theano.config.floatX)
    testX_by_1 = numpy.zeros((100, 3, height, width), dtype=theano.config.floatX)

    trainX_by_2 = numpy.zeros((num_train_set, 3, height / 2, width / 2), dtype=theano.config.floatX)
    valX_by_2 = numpy.zeros((num_validation_set, 3, height / 2, width / 2), dtype=theano.config.floatX)
    testX_by_2 = numpy.zeros((100, 3, height / 2, width / 2), dtype=theano.config.floatX)

    trainX_by_4 = numpy.zeros((num_train_set, 3, height / 4, width / 4), dtype=theano.config.floatX)
    valX_by_4 = numpy.zeros((num_validation_set, 3, height / 4, width / 4), dtype=theano.config.floatX)
    testX_by_4 = numpy.zeros((100, 3, height / 4, width / 4), dtype=theano.config.floatX)

    trainY = numpy.zeros((num_train_set * height * width), numpy.int32)
    valY = numpy.zeros((num_validation_set * height * width), numpy.int32)
    testY = numpy.zeros((100 * height * width), numpy.int32)

    for file in os.listdir('dataset/images_formatted/'):
        if file.endswith('.jpg'):

            fileNumber = file[0: file.find('.')]
            tempLabels = numpy.zeros((height * width), numpy.int32)
            labelPos = 0

            with open('dataset/labels_formatted/' + fileNumber + '.regions.txt', 'r') as labelFile:
                while True:
                    c = labelFile.read(1)
                    if not c:
                        break

                    if c == '-':
                        c = labelFile.read(1)
                        tempLabels[labelPos] = -1
                        labelPos += 1

                    elif c != ' ' and c != '\n':
                        tempLabels[labelPos] = int(c)
                        labelPos += 1

            temp_by_1, temp_by_2, temp_by_4 = get_img_pyramids(misc.imread('dataset/images_formatted/' + file))
            print 'Processed image number ', counter

            if counter < num_train_set:
                trainX_by_1[counter, :, :, :] = temp_by_1
                trainX_by_2[counter, :, :, :] = temp_by_2
                trainX_by_4[counter, :, :, :] = temp_by_4
                trainY[(counter * height * width):((counter + 1) * height * width)] = tempLabels

            elif num_train_set <= counter < num_train_set + num_validation_set:
                valX_by_1[counter - num_train_set, :, :, :] = temp_by_1
                valX_by_2[counter - num_train_set, :, :, :] = temp_by_2
                valX_by_4[counter - num_train_set, :, :, :] = temp_by_4
                valY[((counter - num_train_set) * height * width):(((counter - num_train_set) + 1) * height * width)] = tempLabels

            elif num_train_set + num_validation_set <= counter < num_train_set + num_validation_set + num_test_set:
                testX_by_1[counter - num_train_set - num_validation_set, :, :, :] = temp_by_1
                testX_by_2[counter - num_train_set - num_validation_set, :, :, :] = temp_by_2
                testX_by_4[counter - num_train_set - num_validation_set, :, :, :] = temp_by_4
                testY[((counter - num_train_set - num_validation_set) * height * width):(((counter - num_train_set - num_validation_set) + 1) * height * width)] = tempLabels

            counter += 1

    trainY += 1
    valY += 1
    testY += 1

    numpy.save('dataset/numpy_data/trainX_by_1.npy', trainX_by_1)
    numpy.save('dataset/numpy_data/trainX_by_2.npy', trainX_by_2)
    numpy.save('dataset/numpy_data/trainX_by_4.npy', trainX_by_4)
    numpy.save('dataset/numpy_data/trainY.npy', trainY)

    numpy.save('dataset/numpy_data/valX_by_1.npy', valX_by_1)
    numpy.save('dataset/numpy_data/valX_by_2.npy', valX_by_2)
    numpy.save('dataset/numpy_data/valX_by_4.npy', valX_by_4)
    numpy.save('dataset/numpy_data/valY.npy', valY)

    numpy.save('dataset/numpy_data/testX_by_1.npy', testX_by_1)
    numpy.save('dataset/numpy_data/testX_by_2.npy', testX_by_2)
    numpy.save('dataset/numpy_data/testX_by_4.npy', testX_by_4)
    numpy.save('dataset/numpy_data/testY.npy', testY)

def load_processed_img_data():
    trainX_by_1 = theano.shared(numpy.load('dataset/numpy_data/trainX_by_1.npy'), borrow=True, strict=True)
    trainX_by_2 = theano.shared(numpy.load('dataset/numpy_data/trainX_by_2.npy'), borrow=True, strict=True)
    trainX_by_4 = theano.shared(numpy.load('dataset/numpy_data/trainX_by_4.npy'), borrow=True, strict=True)
    trainY = theano.shared(numpy.load('dataset/numpy_data/trainY.npy'), borrow=True, strict=True)

    valX_by_1 = theano.shared(numpy.load('dataset/numpy_data/valX_by_1.npy'), borrow=True, strict=True)
    valX_by_2 = theano.shared(numpy.load('dataset/numpy_data/valX_by_2.npy'), borrow=True, strict=True)
    valX_by_4 = theano.shared(numpy.load('dataset/numpy_data/valX_by_4.npy'), borrow=True, strict=True)
    valY = theano.shared(numpy.load('dataset/numpy_data/valY.npy'), borrow=True, strict=True)

    testX_by_1 = theano.shared(numpy.load('dataset/numpy_data/testX_by_1.npy'), borrow=True, strict=True)
    testX_by_2 = theano.shared(numpy.load('dataset/numpy_data/testX_by_2.npy'), borrow=True, strict=True)
    testX_by_4 = theano.shared(numpy.load('dataset/numpy_data/testX_by_4.npy'), borrow=True, strict=True)
    testY = theano.shared(numpy.load('dataset/numpy_data/testY.npy'), borrow=True, strict=True)

    return trainX_by_1, trainY, valX_by_1, valY, testX_by_1, testY, trainX_by_2, trainX_by_4, valX_by_2, valX_by_4, testX_by_2, testX_by_4

def generate_color_labels(label):
    height = label.shape[0]
    width = label.shape[1]

    r = numpy.zeros((height, width))
    g = numpy.zeros((height, width))
    b = numpy.zeros((height, width))

    if 0 in label:
        t = numpy.zeros((height, width))
        t[label == 0] = 105
        r += t
        g += t
        b += t

    if 1 in label:
        t = numpy.zeros((height, width))
        t[label == 1] = 176
        r += t

        t = numpy.zeros((height, width))
        t[label == 1] = 196
        g += t

        t = numpy.zeros((height, width))
        t[label == 1] = 222
        b += t

    if 2 in label:
        t = numpy.zeros((height, width))
        t[label == 2] = 210
        r += t

        t = numpy.zeros((height, width))
        t[label == 2] = 105
        g += t

        t = numpy.zeros((height, width))
        t[label == 2] = 30
        b += t

    if 3 in label:
        t = numpy.zeros((height, width))
        t[label == 3] = 255
        r += t

        t = numpy.zeros((height, width))
        t[label == 3] = 20
        g += t

        t = numpy.zeros((height, width))
        t[label == 3] = 147
        b += t

    if 4 in label:
        t = numpy.zeros((height, width))
        t[label == 4] = 139
        r += t

        t = numpy.zeros((height, width))
        t[label == 4] = 0
        g += t

        t = numpy.zeros((height, width))
        t[label == 4] = 139
        b += t

    if 5 in label:
        t = numpy.zeros((height, width))
        t[label == 5] = 0
        r += t

        t = numpy.zeros((height, width))
        t[label == 5] = 255
        g += t

        t = numpy.zeros((height, width))
        t[label == 5] = 255
        b += t

    if 6 in label:
        t = numpy.zeros((height, width))
        t[label == 6] = 0
        r += t

        t = numpy.zeros((height, width))
        t[label == 6] = 255
        g += t

        t = numpy.zeros((height, width))
        t[label == 6] = 0
        b += t

    if 7 in label:
        t = numpy.zeros((height, width))
        t[label == 7] = 255
        r += t

        t = numpy.zeros((height, width))
        t[label == 7] = 69
        g += t

        t = numpy.zeros((height, width))
        t[label == 7] = 0
        b += t

    if 8 in label:
        t = numpy.zeros((height, width))
        t[label == 8] = 218
        r += t

        t = numpy.zeros((height, width))
        t[label == 8] = 165
        g += t

        t = numpy.zeros((height, width))
        t[label == 8] = 32
        b += t

    res = numpy.zeros((height, width, 3))
    res[:, :, 0] = r
    res[:, :, 1] = b
    res[:, :, 2] = g

    return res


from theano.tensor import basic
from theano.gradient import DisconnectedType

class RepeatOp(theano.Op):
    # See the repeat function for docstring

    def __init__(self, axis=None):
        self.axis = axis

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.axis == other.axis)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.axis)

    def make_node(self, x, repeats):
        x = basic.as_tensor_variable(x)
        repeats = basic.as_tensor_variable(repeats)

        # if repeats.dtype not in theano.tensor.discrete_dtypes:
        #     raise TypeError("repeats.dtype must be an integer.")
        #
        # # Some dtypes are not supported by numpy's implementation of repeat.
        # # Until another one is available, we should fail at graph construction
        # # time, not wait for execution.
        # ptr_bitwidth = theano.gof.local_bitwidth()
        # if ptr_bitwidth == 64:
        #     numpy_unsupported_dtypes = ('uint64',)
        # if ptr_bitwidth == 32:
        #     numpy_unsupported_dtypes = ('uint32', 'int64', 'uint64')
        #
        # if repeats.dtype in numpy_unsupported_dtypes:
        #     raise TypeError(
        #         ("dtypes %s are not supported by numpy.repeat "
        #          "for the 'repeats' parameter, "
        #          % str(numpy_unsupported_dtypes)), repeats.dtype)
        #
        # if self.axis is None:
        #     broadcastable = [False]
        # else:
        #     try:
        #         const_reps = basic.get_scalar_constant_value(repeats)
        #     except basic.NotScalarConstantError:
        #         const_reps = None
        #     if const_reps == 1:
        #         broadcastable = x.broadcastable
        #     else:
        #         broadcastable = list(x.broadcastable)
        #         broadcastable[self.axis] = False

        out_type = theano.tensor.ftensor4

        return theano.Apply(self, [x, repeats], [out_type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        repeats = inputs[1]
        z = output_storage[0]

        if self.axis is 2:
            repeat_style = (1, 1, repeats, 1)
        else:
            repeat_style = (1, 1, 1, repeats)

        z[0] = scipy.ndimage.zoom(input=x, zoom=repeat_style, order=1)

    def connection_pattern(self, node):

        return [[True], [False]]

    def grad(self, (x, repeats), (gz, )):
        if repeats.ndim == 0:
            if self.axis is None:
                axis = x.ndim
            else:
                if self.axis >= 0:
                    axis = self.axis + 1
                else:
                    axis = self.axis + x.ndim + 1

            shape = [x.shape[k] for k in range(x.ndim)]
            shape.insert(axis, repeats)

            return [gz.reshape(shape, x.ndim + 1).sum(axis=axis),
                    DisconnectedType()()]
        elif repeats.ndim == 1:
            # For this implementation, we would need to specify the length
            # of repeats in order to split gz in the right way to sum
            # the good part.
            raise NotImplementedError()
        else:
            raise ValueError()

    def infer_shape(self, node, ins_shapes):
        i0_shapes = ins_shapes[0]
        repeats = node.inputs[1]
        out_shape = list(i0_shapes)

        # uint64 shape are not supported.
        dtype = None
        if repeats.dtype in ['uint8', 'uint16', 'uint32']:
            dtype = 'int64'
        if self.axis is None:
            if repeats.ndim == 0:
                if len(i0_shapes) == 0:
                    out_shape = [repeats]
                else:
                    res = 1
                    for d in i0_shapes:
                        res = res * d
                    out_shape = (res * repeats, )
            else:
                out_shape = [theano.tensor.sum(repeats, dtype=dtype)]
        else:
            if repeats.ndim == 0:
                out_shape[self.axis] = out_shape[self.axis] * repeats
            else:
                out_shape[self.axis] = theano.tensor.sum(repeats, dtype=dtype)
        return [out_shape]

    def __str__(self):
        return self.__class__.__name__


def interpolate_tensor(x, repeats, axis=None):
    repeats = theano.tensor.as_tensor_variable(repeats)

    # if repeats.ndim > 1:
    #     raise ValueError('The dimension of repeats should not exceed 1.')

    # if repeats.ndim == 1:
    return RepeatOp(axis=axis)(x, repeats)
    # else:
    #     if axis == None:
    #        axis = 0
    #        x = x.flatten()
    #     else:
    #         if axis >= x.ndim:
    #             raise ValueError('Axis should not exceed x.ndim-1.')
    #         if axis < 0:
    #             axis = x.ndim+axis
    #
    #     shape = [x.shape[i] for i in xrange(x.ndim)]
    #
    #     # shape_ is the shape of the intermediate tensor which has
    #     # an additional dimension comparing to x. We use alloc to
    #     # allocate space for this intermediate tensor to replicate x
    #     # along that additional dimension.
    #     shape_ = shape[:]
    #     shape_.insert(axis+1, repeats)
    #
    #     # shape is now the shape of output, where shape[axis] becomes
    #     # shape[axis]*repeats.
    #     shape[axis] = shape[axis]*repeats
    #
    #     # dims_ is the dimension of that intermediate tensor.
    #     dims_ = list(numpy.arange(x.ndim))
    #     dims_.insert(axis+1, 'x')
    #
    #     # After the original tensor is duplicated along the additional
    #     # dimension, we reshape it to the expected output shape, and
    #     # return the output z.
    #     z = theano.tensor.alloc(x.dimshuffle(*dims_), *shape_).reshape(shape)
    #     return z

import time
if __name__ == '__main__':
    # pre_process_img_and_labels_multi_scale(
    #     height=240,
    #     width=320,
    #     num_train_set=500,
    #     num_validation_set=100,
    #     num_test_set=100,
    # )
    #
    # train_set_x_by_1, train_set_y, valid_set_x_by_1, valid_set_y, test_set_x_by_1, test_set_y, train_set_x_by_2, \
    # train_set_x_by_4, valid_set_x_by_2, valid_set_x_by_4, test_set_x_by_2, test_set_x_by_4 \
    #     = load_processed_img_data()
    #
    # print train_set_x_by_1.get_value()

    y = numpy.random.random_sample((5, 512, 60, 80))
    y = numpy.asarray(y, dtype=theano.config.floatX)
    x = theano.shared(value=y, strict=True)
    # print x.eval()
    start = time.clock()
    x = interpolate_tensor(x, 4, axis=3)
    x = interpolate_tensor(x, 4, axis=2)
    print time.clock() - start

    # p = scipy.ndimage.zoom(y, (1, 1, 4, 4), order=3)
    # print p

    x = theano.shared(value=y, strict=True)
    start = time.clock()
    x = theano.tensor.repeat(x, 4, axis=3)
    x = theano.tensor.repeat(x, 4, axis=2)
    print time.clock() - start