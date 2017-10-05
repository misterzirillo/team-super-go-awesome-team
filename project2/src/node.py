from abc import ABCMeta, abstractmethod

'''
Abstract class for node
'''
class Node(metaclass=ABCMeta):

    # the one thing a node does
    @abstractmethod
    def process(self, inputVector, inputWeights):
        pass