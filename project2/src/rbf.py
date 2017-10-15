import math
import numpy
import functools
from itertools import permutations, combinations, product

class RBF:

	#set the number of inputs, the number of hidden nodes, and the number of outputs
	def __init__(self, k, outputs, trainingData):

		self.trainingData = numpy.asarray(trainingData)

		# the argument dimension > 1
		self.n = len(trainingData[0])

		#the output layer, will be 1 node for our purposes
		self.outputs = outputs

		#set some tunable parameters that may or may not get adjusted
		self.eta = .5

		self.centers = list(map(numpy.asarray, centers(k, self.n)))
		self.k = len(self.centers)
		print(k)

		# determine max distance each center to all other centers
		maxCenterDistance = distance(numpy.array([-3] * self.n), numpy.array([3] * self.n))
		self.sigma = maxCenterDistance / (2 * self.k)

		# get some weights boy
		self.weights = numpy.random.sample(k)

		print(len(self.weights))
		

    #one pass of gradient descent on the weights from the hidden layer to the output
	def train(self):
		#in general compare the output to the actual and adjust using the adaline learning rule
		#Weights = Weights + eta(actual value at input - ouput of the network at input)* input vector
		#run until it converges to the least squared error which is E = (actual value - output of network)^2 

		targetError = .2
		trainingBatches = numpy.split(self.trainingData, 5)

		trainingIndex = 0
		iterations = 0
		allBatchError = []
		allForwardPropError = []

		m = len(self.trainingData)

		with open('error_rates.csv', 'w') as f:
			while iterations < 100:
				epockavg = []
				for data in self.trainingData:
					#print('_________ITERATION________')
					#print(data)
					inputVector = data[0]
					#print(inputVector)
					expectedOutput = data[1]
					#print(expectedOutput)
					networkOutput = self.propagate(inputVector)
					#print(networkOutput)

					xTrans = networkOutput[1].transpose()

					hlo = networkOutput[1]

					hloT = hlo.transpose()
					hypothesis = networkOutput[0]
					loss = hypothesis - expectedOutput
					# avg cost per example (the 2 in 2*m doesn't really matter here.
					# But to be consistent with the gradient, I include it)
					cost = numpy.sum(loss ** 2) / (2 * m)
					#print("Iteration %d | Cost: %f" % (i, cost))
					# avg gradient per example
					gradient = numpy.dot(xTrans, loss) / m

					#weightUpdate = self.eta * error * networkOutput[1]
					oldWeight = self.weights
					self.weights = self.weights - self.eta * gradient

					#print(oldWeight)
					#print(weightUpdate)
					#print(self.weights)
					#print()

					#if math.isnan(error): return
					#print(error)
					epockavg.append(loss)
					#batchError.append(error)

				recordMe = numpy.mean(epockavg)
				allForwardPropError.append(recordMe)
				print('epc ' + str(iterations) + ':' + str(recordMe))
				iterations += 1
		
		return allForwardPropError

		# STOP TRAINING RIGHT WHEN CONVERGE??

		# for epoch in trainingBatches:
		# 	iterations = iterations + 1
		# 	notConverged = True

		# 	while iterations < 20:
		# 		batchError = []
		# 		batchWeights = []

				# iterate over data in batch and collect error for each example
				

					#itrWeights = self.weights + (self.eta * error * networkOutput[1])



					#print(itrWeights)
				# 	batchWeights.append(itrWeights)
				# 	print(error)

				# print('________EPOCH END__________')
				# #print(batchWeights)
				# newWeights = numpy.mean(numpy.asarray(batchWeights), axis=0)
				# #print(newWeights)
				# #oldWeights = self.weights
				# #self.weights = self.weights - newWeights
				# #print(self.weights)
				# meanError = numpy.mean(batchError)
				# allBatchError.append(meanError)
				# iterations += 1
				

	def propagate(self, input):

		#pass all the inputs to the k hidden nodes		
		hiddenLayerOutput = numpy.asarray(list(map(lambda c: guassian(input, c, self.sigma), self.centers)))
		thing = hiddenLayerOutput * self.weights
		result = sum(thing)
		return (result, hiddenLayerOutput)

def distance(x, y):
	return numpy.linalg.norm(x - y)

#the activation function of the hidden nodes
def guassian(input, center, sigma):
	#apply this function to the weighted sum at each hidden node
	#it will be the hyperbolic tangent function in our case
	dist = distance(input, center)

	return math.exp(-1 * ((dist**2) / (2 * sigma**2)))


def centers(k, argDimension):

	if k < 2: raise ValueError('k < 2')
	if argDimension < 1: raise ValueError('argDimension < 1')

	kRoot = k**(1/float(argDimension))
	gridInterval = float(6) / (kRoot - 1)
	#print(gridInterval)

	gridStep = [-3.0]
	it = 1
	while (-3 + it * gridInterval < 3):
		gridStep.append(-3 + it * gridInterval)
		it = it + 1
	gridStep.append(3.0)


	#print(gridStep)
	#print(len(gridStep))
	perms = list(product(gridStep, repeat=argDimension))
	#print(len(perms))
	#print(perms)

	return perms

if __name__ == '__main__':
	#the constructor builds the network according to the arguments
	#k means is called to determine the number of hidden nodes and sigma
	#the constructor randomizes the centers and initial weights.
	#enter a loop until the training method converges
	#propagate to produce the output
	#the end

	#Notes:
	#we want to use stochastic updating
	pass
