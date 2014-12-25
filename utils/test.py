__author__ = 'mateuszopala'
import numpy as np
import theano


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(w):
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1)
    return dist


def random_array(size, eps=1.0):
    return np.asarray(
        np.random.uniform(
            low=-eps,
            high=eps,
            size=size
        ), dtype=theano.config.floatX)


def gaussian_array(size, mean, std):
    return np.asarray(np.random.normal(mean, std, size), dtype=theano.config.floatX)