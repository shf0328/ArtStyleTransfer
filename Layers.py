import theano
import theano.tensor as T
from theano.tensor.signal import pool
import numpy


class InputLayer():
    def __init__(self, incoming, shape):
        self.output = incoming
        self.outputShape = shape
        self.params = None
        self.nextLayers = []

    def propagate_output(self):
        for layer in self.nextLayers:
            layer.propagate_output()


class ConvLayer():
    def __init__(self, incomingLayer, num_filters, filter_size, rng, flip_filters=False):
        # link layers
        self.incomingLayer = incomingLayer
        self.input = incomingLayer.output
        self.inputShape = incomingLayer.outputShape
        self.incomingLayer.nextLayers.append(self)
        self.nextLayers = []
        # initial params
        self.filter_shape = (num_filters, self.inputShape[1], filter_size, filter_size)
        fan_in = numpy.prod(self.filter_shape[1:])
        fan_out = (self.filter_shape[0] * numpy.prod(self.filter_shape[2:]) // numpy.prod([1, 1]))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        b_values = numpy.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        # keep track of them
        self.params = [self.W, self.b]
        # prepare output
        self.output = None
        self.outputShape = None
        # convolve input feature maps with filters
        self.filter_flip = flip_filters
        self.update_output()

    def update_output(self):
        self.input = self.incomingLayer.output
        self.inputShape = self.incomingLayer.outputShape
        # convolve input feature maps with filters
        conv_out = T.nnet.conv2d(
            input=self.input,
            filters=self.W,
            filter_shape=self.filter_shape,
            input_shape=self.inputShape,
            border_mode='half',
            filter_flip=self.filter_flip,
        )
        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.outputShape = (self.inputShape[0], self.filter_shape[0], self.inputShape[2], self.inputShape[3])

    def propagate_output(self):
        self.update_output()
        for layer in self.nextLayers:
            layer.propagate_output()

class PoolLayer():
    def __init__(self, incomingLayer, poolsize, mode):
        # link layer
        self.incomingLayer = incomingLayer
        self.input = incomingLayer.output
        self.inputShape = incomingLayer.outputShape
        self.incomingLayer.nextLayers.append(self)
        self.nextLayers = []
        self.params = None
        # prepare pool
        self.mode = mode
        self.poolsize = poolsize
        self.output = None
        self.outputShape = None
        # calc pool
        self.update_output()

    def update_output(self):
        self.input = self.incomingLayer.output
        self.inputShape = self.incomingLayer.outputShape
        pooled_out = pool.pool_2d(
            input=self.input,
            ds=self.poolsize,
            ignore_border=True,
            mode=self.mode
        )
        self.output = pooled_out
        self.outputShape = (self.inputShape[0], self.inputShape[1], self.inputShape[2] // 2, self.inputShape[3] // 2)

    def propagate_output(self):
        self.update_output()
        for layer in self.nextLayers:
            layer.propagate_output()

