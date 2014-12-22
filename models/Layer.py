__author__ = 'mateuszopala'
import theano
import theano.tensor as T
"""
Each of models should inherit from Layer and either Unsupervised or Supervised.
"""


class Layer(object):
    """
    Layer is base class for all bricks. Each of your models should inherit from Layer and override at least three
    methods:
    - initialize    - Performs model initialization
    - parameters    - Returns list of model parameters
    - t_output      - Theano function that returns output (activation)
    """

    def initialize(self, *args, **kwargs):
        pass

    def parameters(self):
        pass

    def t_output(self, t_data, t_parameters):
        pass

    def output(self):
        t_data = T.matrix('data')
        t_function = theano.function(inputs=[t_data],
                                     outputs=self.t_output(t_data, self.parameters()))
        return t_function

    @staticmethod
    def _t_number_of_examples(t_data):
        return T.cast(T.shape(t_data)[0], theano.config.floatX)


class Unsupervised(object):
    """
            @t_gradient()      returns gradient w.r.t. to parameters, but as theano variables
            @t_cost()          returns cost/loss for given data array. Cost function doesn't have to be objective function
                               (which is being minimized). E.g, in restricted boltzmann machine, one can define t_cost as
                               negative energy (objective function) as well as reconstruction error
    """

    def t_gradient(self, t_data, t_parameters):
        cost = self.t_cost(t_data, t_parameters)
        return [T.grad(cost, t_param, disconnected_inputs='warn') for t_param in t_parameters]

    def t_cost(self, t_data, t_parameters):
        pass

    def parameters(self):
        pass

    def cost(self):
        t_data = T.matrix('data')
        t_function = theano.function(inputs=[t_data],
                                     outputs=self.t_cost(t_data, self.parameters()))
        return t_function


class Supervised(object):
    def t_gradient(self, t_data, t_target, t_parameters):
        cost = self.t_cost(t_data, t_target, t_parameters)
        return [T.grad(cost, t_param, disconnected_inputs='warn') for t_param in t_parameters]

    def t_cost(self, t_data, t_target, t_parameters):
        pass

    def parameters(self):
        pass

    def cost(self):
        t_data = T.matrix('data')
        t_target = T.matrix('target')
        t_function = theano.function(inputs=[t_data, t_target],
                                     outputs=self.t_cost(t_data, t_target, self.parameters()))
        return t_function


# TODO: implement layer serialization
class Pickleable(object):
    pass