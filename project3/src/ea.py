import argparse
import math
from datetime import datetime
from abc import ABC, abstractmethod
import collections
import numpy as np
from mlpnetwork import MLPNetwork
import helpers

class EA(ABC):
	
	pop = []
	bestEva=[]
	validOne = []
	
	def __init__(self, shape, mu):
		#self.transfer = transfer
		self.shape=shape
		self.mu = mu
		self.initializePop(mu)
		
		
	@abstractmethod
	def train(self):
		self.trainingErrors = []
		self.validationErrors =[]

	#display structure stats
	def check(self):
		print("Network shape: "+ str(self.shape) + "\n"+ "Individual Length: " +  str(len(self.pop[1])) + "\n"+ "First five weights: " + str(self.pop[1][0:5]) + "\n")

	#display fitness rankings
	def checkFit(self):
		for i in range(len(self.sortFit)):
			print(self.sortFit[i])
		
	#generate a random population
	def initializePop(self, mu):
		self.builtNetworks = [MLPNetwork(self.shape) for whatever in range(mu)]
		self.pop = list(map(lambda net: self.cereal(net.weights), self.builtNetworks))
		  
	#evaluate the fitness of the population on some loss function
	def evaluateFitness(self, individual, x, y): 

		# maybe optimize this later with pre-created networks
		rehydrated = MLPNetwork(self.shape)
		rehydrated.weights = self.uncereal(individual)

		hypothesis = rehydrated.propagate(x)

		return helpers.percentCorrect(hypothesis, y)
	
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
		for layer, nextLayer in zip(self.shape[:-1], self.shape[1:]): 
			numWeights = (layer+1) * nextLayer
			newLastIndex = lastIndex + numWeights
			weightsForLayer = np.reshape(np.array(arr[lastIndex:newLastIndex]), [nextLayer, layer+1]) 
			acc.append(weightsForLayer) 
			lastIndex = newLastIndex
		return acc

	def postIterationProcess(self, x, y, fit, t):
		trainingError = 100 - fit        
		validationError = 100 - self.evaluateFitness(self.pop[-1], x, y)

		#print('training error: ' + str(trainingError) + '\tvalidation error: ' + str(validationError))

		self.trainingErrors.append(trainingError)
		self.validationErrors.append(validationError)
		converged = False
		#converged = self.validationErrors[-1] > self.validationErrors[-2] if len(self.validationErrors) > 10 else False
		if t > 10:
			converged = self.validationErrors[-1] > np.mean(self.validationErrors[10:]) + np.std(self.validationErrors[10:])
			if converged:
				print("Convergence check reached at Generation " + str(t))
				print(str(self.validationErrors[-1]) + ' > ' + str(np.mean(self.validationErrors)) + str(np.std(self.validationErrors)))

		return converged

