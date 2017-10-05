from abc import ABCMeta, abstractmethod

"""
Abstraction of what we consider a network to be. 
To implement do:

from network import Network
class Concrete(Network):
    # implement methods
"""
class Network(metaclass=ABCMeta):

    # this method should initialize the layers
    # and the nodes within those layers for the
    # implementation-specific network
    @abstractmethod
    def connect(self):
        pass

    # this method should handle the forward 
    # propagation of input though all layers
    # and return the output of the last layer.
    @abstractmethod
    def propagate(self, input):
        pass

    # this method should train the network given
    # a dataset. will contain calles to propogate
    # and the implementation of backpropogation
    @abstractmethod
    def train(self, dataset):
        pass
        