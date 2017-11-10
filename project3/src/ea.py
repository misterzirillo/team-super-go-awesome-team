import argparse
import math
from datetime import datetime
from abc import ABC, abstractmethod
import collections
import numpy as np
from mlpnetwork import MLPNetwork

class EA(ABC):
    
    pop = []
    numWeights =0
    
    def __init__(self, shape, mu):
        #self.transfer = transfer
        self.shape=shape
        self.mu = mu
        self.initializePop(mu)

        self.trueShape = [n + 1 for n in shape[:-1]]
        self.trueShape.append(self.shape[-1])
        
        
    @abstractmethod
    def train(self):
        pass
    
        #generate a random population
    def initializePop(self, mu):
        self.builtNetworks = [MLPNetwork(self.shape) for whatever in range(mu)]
        self.pop = list(map(lambda net: self.cereal(net.weights), self.builtNetworks))
          
    #evaluate the fitness of the population on some loss function
    def evaluateFitness(self, individual, x, y): 
        correctIndex = y.index(max(y)) # [0, 0, 1] -> 2

        network = MLPNetwork(self.shape)

        hypothesis = individual.propagate(x)
        hypothesizedIndex = hypothesis.index(max(hypothesis))
        return sum(correctIndex == hypothesizedIndex) / len(y or x)
    
    #select the parents from the population
    @abstractmethod
    def selectFrom(self):
        pass

    #generate offspring according to the crossover rate
    #global, uniform
    # should return a whole new copy of the population
    # example: self.pop = self.crossOver()
    @abstractmethod
    def crossOver(self):
        pass

    #mutate the offspring according to the mutatation rate
    # should return a whole new copy of the population
    # example: self.pop = self.mutate()
    @abstractmethod
    def mutate(self):
        pass
    
    #take a weight matrix and represent it as a string
    def cereal(self, x):
        if isinstance(x, collections.Iterable):
            return [a for i in x for a in self.cereal(i)]
        else:
            return [x]

    def uncereal(self, arr): 
        lastIndex = 0 
        acc = [] 
        for layer, nextLayer in zip(self.trueShape[:-1], self.trueShape[1:]): 
            numWeights = layer * nextLayer 
            newLastIndex = lastIndex + numWeights 
            weightsForLayer = np.reshape(np.array(arr[lastIndex:newLastIndex]), [nextLayer, layer]) 
            acc.append(weightsForLayer) 
            lastIndex = newLastIndex
        return acc
