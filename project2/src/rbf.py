import math
import numpy as np
import functools
import argparse
from itertools import product
from datetime import datetime
from main import readDatasetFromFile

class RBF:

	#set the number of inputs, the number of hidden nodes, and the number of outputs
	def __init__(self, k, trainingData, eta):

		self.trainingData = np.asarray(trainingData)

		# the argument dimension > 1
		self.n = len(trainingData[0][0])

		#set some tunable parameters that may or may not get adjusted
		self.eta = eta

		self.centers = list(map(np.asarray, centers(k, self.n)))
		self.k = len(self.centers)

		# determine max distance each center to all other centers
		maxCenterDistance = distance(np.array([-3] * self.n), np.array([3] * self.n))
		self.sigma = maxCenterDistance / math.sqrt(2 * self.k)

		# get some weights boy
		self.weights = np.ones(self.k)
		

    #one pass of gradient descent on the weights from the hidden layer to the output
	def train(self):
		#in general compare the output to the actual and adjust using the adaline learning rule
		#Weights = Weights + eta(actual value at input - ouput of the network at input)* input vector
		#run until it converges to the least squared error which is E = (actual value - output of network)^2 

		batchSize = 2 ** (self.n + 2)

		timelimitMinutes = 30
		batchMSEs =[]
		epoch = 0
		notConverged = True
		epochLimit = 1000
		weightChangeLimiter = 0

		printcount = 0

		start = datetime.now()
		def checkTimeMinutes():
			return (datetime.now() - start).total_seconds() // 60
		
		try:
			while notConverged and epoch < epochLimit and checkTimeMinutes() < timelimitMinutes:

				batches = get_mini_batches(self.trainingData, batchSize)

				for i, batch in enumerate(batches):
					inputs, actuals = zip(*batch)
					X = np.asarray(list(map(self.propagateToHiddenLayer, inputs)))
					weightedSums = X.dot(self.weights) # activation function of output node
					m = len(batch)
					gradient, error = doGDForBatch(X, weightedSums, actuals, m)

					# update weights
					self.weights = self.weights - self.eta * gradient

					# some accounting
					printcount += m
					if printcount > 1000:
						batchError = np.mean([e ** 2 for e in error])
						print('epoch\t{}\tbatch\t{}\tmins_elapsed\t{}\terror_mse\t{}'
								.format(epoch, i, checkTimeMinutes(), batchError))
						batchMSEs.append(batchError)

				epoch += 1
				
		except KeyboardInterrupt:
			pass
		
		return batchMSEs
				

	def propagateToHiddenLayer(self, input):
		#pass all the inputs to the k hidden nodes
		# and computer gaussian for each node
		hiddenLayerOutput = np.fromiter(map(lambda c: guassian(input, c, self.sigma), self.centers), np.float)
		return hiddenLayerOutput


def doGDForBatch(x, h, y, m):
	error = h - y
	gradient = x.T.dot(error) / m
	return (gradient, error)

def distance(x, y):
	return np.linalg.norm(x - y)

#the activation function of the hidden nodes
def guassian(input, center, sigma):
	#apply this function to the weighted sum at each hidden node
	#it will be the hyperbolic tangent function in our case
	dist = distance(input, center)

	return math.exp(-1 * ((dist**2) / (2 * sigma**2)))

# given a dataset return minibatches 
def get_mini_batches(dataset, batch_size):
	random_idxs = np.random.choice(len(dataset), len(dataset), replace=True)
	X_shuffled = list(map(lambda i: dataset[i], random_idxs))
	mini_batches = [X_shuffled[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
	return mini_batches

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

	perms = list(product(gridStep, repeat=argDimension))
	
	return perms

if __name__ == '__main__':
	# make parser
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset', help='The name of the dataset in ../data to use')
	parser.add_argument('numCentersK', type=int, help='The target number of centers')
	args = parser.parse_args()

	data = readDatasetFromFile(args.dataset)
	k = args.numCentersK
	eta = .1

	network = RBF(k, data, eta)
	error = network.train()

	netName = 'rbf-n{}-k{}'.format(network.n, network.k)
	print('finished training network ' + netName)

	errorFile = netName + '-error.csv'
	print('printing MSE epoch values to ' + errorFile)	
	with open(errorFile, 'w') as f:
		f.writelines('\n'.join(map(str, error)))
	
	summaryFile = netName + '-summary.txt'
	print('printing network summary to ' + summaryFile)
	with open(summaryFile, 'w') as f:
		f.write('n=' + str(network.n) + '\n')
		f.write('centers=' + str(network.centers) + '\n')
		f.write('weights=' + np.array2string(network.weights) + '\n')
