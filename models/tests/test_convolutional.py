__author__ = 'mateuszopala'
import unittest
from utils.test import *
from ..convolutional import Convolutional
from scipy.signal import convolve2d
from numpy.testing import assert_almost_equal


class TestConvolutional(unittest.TestCase):
    BATCH_SIZE = 1
    STACK_SIZE = 1
    ROWS = 8
    COLS = 8
    FILTER_SHAPE = (1, STACK_SIZE, 3, 3)

    def setUp(self):
        input_shape = self.BATCH_SIZE, self.STACK_SIZE, self.ROWS, self.COLS
        self.data = random_array(input_shape)
        self.conv_layer = Convolutional(input_shape, self.FILTER_SHAPE)
        self.conv_layer.initialize()

    def test_output(self):
        weights, bias = map(lambda x: x.get_value(), self.conv_layer.parameters())
        theano_value = self.conv_layer.output()(self.data)
        numpy_value = convolve2d(self.data.reshape((self.ROWS, self.COLS)),
                                 weights.reshape((self.FILTER_SHAPE[2], self.FILTER_SHAPE[3])), mode='valid') + bias
        numpy_value = sigmoid(numpy_value)
        # TODO: implement max-pooling for numpy. TEST is correct without pooling
        assert_almost_equal(theano_value.flatten(), numpy_value.flatten())
