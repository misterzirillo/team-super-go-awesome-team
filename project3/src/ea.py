import argparse
import math
from datetime import datetime
from abc import ABC, abstractmethod
import collections
import numpy as np

class EA(ABC):
    
    pop = []
    numWeights =0
    
    def __init__(self, shape, mu):
        self.shape=shape
        self.mu = mu
        self.initializePop(mu)
        self.numWeights = sum([self.shape[i] * self.shape(i + 1) for i in range(len(self.shape - 1))])
        
        
    @abstractmethod
    def train(self, data, maxGenerations = 1000000):
        pass
    
        #generate a random population
    def initializePop(self, mu):
        self.pop = [np.random.uniform(size=numWeights) for i in range(mu)]
          
    #evaluate the fitness of the population on some loss function
    def evaluateFitness(self, individual, x, y): 
        correctIndex = y.index(max(y)) # [0, 0, 1] -> 2
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
    def cereal(self, uncerealWeights):
        if isinstance(x, collections.Iterable):
            return [a for i in x for a in flatten(i)]
        else:
            return [x]

    def uncereal(arr, shape): 
        lastIndex = 0 
        acc = [] 
        for layer, nextLayer in zip(shape[:-1], shape[1:]): 
            numWeights = layer * nextLayer 
            newLastIndex = lastIndex + numWeights 
            weightsForLayer = np.reshape(np.array(arr[lastIndex:newLastIndex]), [nextLayer, layer]) 
            acc.append(weightsForLayer) 
            lastIndex = newLastIndex
        return acc
