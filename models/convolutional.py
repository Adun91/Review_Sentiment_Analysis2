__author__ = 'mateuszopala'

from models.layer import Layer
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams
from functools import *
import numpy as np


class Convolutional(Layer):
    def __init__(self, input_shape, filter_shape, pool_size=(2, 2), activation=T.nnet.sigmoid):
        self.is_initialized = False
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.pool_size = pool_size
        self.activation = activation
        self.ts_weights, self.ts_bias = None, None
        assert input_shape[1] == filter_shape[1]
        self.fan_in = reduce(lambda x, y: x * y, self.filter_shape[1:], 1)

    def initialize(self, weights=None, bias=None, seed=None, force=False):
        if self.is_initialized and not force:
            return
        if seed:
            np.random.seed(seed)
            self.theano_rng = RandomStreams(seed)
        if weights is None:
            # Deeplearning.net tutorial initialization - we should check out different ways
            weights = np.asarray(np.random.uniform(
                low=-np.sqrt(3. / self.fan_in),
                high=np.sqrt(3. / self.fan_in),
                size=self.filter_shape), dtype=theano.config.floatX)

        if bias is None:
            bias = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)

        # Maybe we should check if shared variables were already defined, and use set_value instead (?? buska's comment
        # i guess, about memory leak i guess, i guess he's right - Mateusz)
        self.ts_weights = theano.shared(value=weights, name='weights', borrow=True)
        self.ts_bias = theano.shared(value=bias, name='bias', borrow=True)

        self.is_initialized = True

    def parameters(self):
        if not self.is_initialized:
            # TODO: change exception type
            raise Exception("Model not initialized!")
        return self.ts_weights, self.ts_bias

    def t_output(self, t_data, t_parameters):
        t_weights, t_bias = t_parameters
        conv_out = conv2d(t_data, t_weights, image_shape=self.input_shape, filter_shape=self.filter_shape)
        pooled_out = max_pool_2d(conv_out, self.pool_size, ignore_border=True)
        return self.activation(pooled_out + t_bias.dimshuffle('x', 0, 'x', 'x'))