import math
import numpy as np
import functools
import argparse
from itertools import product
from datetime import datetime
from main import readDatasetFromFile, fileRelativeToHere
import re

# represents an instance of a RBF neural network
class RBF:

	#set the number of inputs, the number of hidden nodes, and the number of outputs
	def __init__(self, proposedK, trainingData, eta, n=None):

		self.trainingData = np.asarray(trainingData)

		# the argument dimension > 1
		if trainingData:
			self.n = len(trainingData[0][0])
		else:
			self.n = n

		#set some tunable parameters that may or may not get adjusted
		self.eta = eta # learning rate
		self.centers = list(map(np.asarray, centers(proposedK, self.n)))
		self.k = len(self.centers)

		# determine max distance each center to all other centers
		maxCenterDistance = distance(np.array([-3] * self.n), np.array([3] * self.n))
		self.sigma = maxCenterDistance / math.sqrt(2 * self.k)

		# get some weights boy
		self.weights = np.ones(self.k)
		

    # iteratively train the network using the training data passed in to the constructor
	# training will run for a maximum of 1000 iterationss of the whole dataset or 30 minutes
	# uses a mini-batch update methodology
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
				
	# this method returns an array of the outputs from each radial basis function
	def propagateToHiddenLayer(self, input):
		#pass all the inputs to the k hidden nodes
		# and computer gaussian for each node
		hiddenLayerOutput = np.fromiter(map(lambda c: guassian(input, c, self.sigma), self.centers), np.float)
		return hiddenLayerOutput

	# helper to analyze MSE over a dataset that is parsed via the main.readDatasetFromFile function
	def doTest(self, dataset):
		inputs, actuals = zip(*dataset)
		X = np.asarray(list(map(self.propagateToHiddenLayer, inputs)))
		weightedSums = X.dot(self.weights) # activation function of output node
		m = len(dataset)
		_, error = doGDForBatch(X, weightedSums, actuals, m)

		mse = np.mean([e**2 for e in error])
		return mse

''' HELPERS '''

# batch gradient descent method for input vector x, hypothesis vector h, actual value y, and data size m
def doGDForBatch(x, h, y, m):
	error = h - y
	gradient = x.T.dot(error) / m
	return (gradient, error)

# distance between vectors
def distance(x, y):
	return np.linalg.norm(x - y)

#the activation function of our hidden nodes
def guassian(input, center, sigma):
	#apply this function to the weighted sum at each hidden node
	#it will be the hyperbolic tangent function in our case
	dist = distance(input, center)

	return math.exp(-1 * ((dist**2) / (2 * sigma**2)))

# given a dataset, return minibatches using a random sampling with replacement mechanism
def get_mini_batches(dataset, batch_size):
	random_idxs = np.random.choice(len(dataset), len(dataset), replace=True)
	X_shuffled = list(map(lambda i: dataset[i], random_idxs))
	mini_batches = [X_shuffled[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
	return mini_batches

# given a desired number of centers k and the dimension of the input vector
# return the centers that will represent each radial basis function node in
# our network. We want the centers to be laid out in a grid fashion, so the
# number of centers returned may differ from the proposed k.
def centers(proposedK, argDimension):

	if proposedK < 2: raise ValueError('k < 2')
	if argDimension < 1: raise ValueError('argDimension < 1')

	# find the number of grid lines for each "side" of the input space
	kRoot = proposedK**(1/float(argDimension))

	# divide the input space along those grid lines
	gridInterval = float(6) / (kRoot - 1)

	# get the coordinates of each grid line from -3 to 3 (interesting points in rosenbrock)
	gridStep = [-3.0]
	it = 1
	while (-3 + it * gridInterval < 3):
		gridStep.append(-3 + it * gridInterval)
		it = it + 1
	gridStep.append(3.0)

	# permute grid line values to get every point in the grid
	perms = list(product(gridStep, repeat=argDimension))
	
	return perms

# re-create a saved network
def loadNetwork(networkSummaryFilename):

	summaryFile = fileRelativeToHere('../nets/' + networkSummaryFilename)
	allLines = []
	with open(summaryFile, 'r') as f:
		for l in f:
			allLines.append(l)

	# parse weights
	weightstr = ''.join(allLines[2:]).split('=')[1]
	weightstr = re.sub('\n', '', weightstr)
	weightstr = re.sub('\s+', ',', weightstr)
	weightstr = re.sub(',', '', weightstr, count=1)

	n = eval(allLines[0].split('=')[1].lstrip())
	weights = np.array(eval(weightstr))
	k = len(weights)

	network = RBF(k, None, 0.1, n=n)
	network.weights = weights

	return network

# usage: python rbf.py 'dataset-name-from-data-folder' proposed-K
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
