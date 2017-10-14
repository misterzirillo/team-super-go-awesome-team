import math
import numpy
import functools

class RBF:

	#set the number of inputs, the number of hidden nodes, and the number of outputs
	def __init__(self, k, outputs, trainingData):

		# the argument dimension > 1
		self.n = len(inputs)

		# number of hidden nodes
		self.k = k

		#the output layer, will be 1 node for our purposes
		self.outputs = outputs

		#set some tunable parameters that may or may not get adjusted
		self.eta = .1

        #randomize the initial centers of the hidden nodes
		self.centers = numpy.random.choice(trainingData.keys(), k)

		# determines max distance of one x to all centers
		def maxDistanceToCenters(x):
			return max(map(centers, lambda y: distance(numpy.asarray(x), numpy.asarray(y))))

		# determine max distance each center to all other centers
		self.sigma = max(map(maxDistanceToCenters, centers))

		# get some weights boy
		self.weights = numpy.random.sample(k)
		

    #one pass of gradient descent on the weights from the hidden layer to the output
	def train(self):
		#in general compare the output to the actual and adjust using the adaline learning rule
		#Weights = Weights + eta(actual value at input - ouput of the network at input)* input vector
		#run until it converges to the least squared error which is E = (actual value - output of network)^2 

		trainingBatches = numpy.split(nupmy.asarray(list(self.trainingData.keys())), len(self.trainingData) / 100)

		trainingIndex = 0
		iterations = 0
		allBatchError = []
		allForwardPropError = []

		for batch in trainingBatches:
			iterations = iterations + 1
			notConverged = True

			while notConverged:
				batchError = []

				# iterate over data in batch and collect error for each example
				for data in batch:
					inputVector = tuple(data)
					expectedOutput = self.trainingData[inputVector]
					networkOutput = self.propagate(inputVector)
					error = expectedOutput - networkOutput
					allForwardPropError.append(error)
					batchError.append(error)

				meanError = numpy.mean(batchError)
				allBatchError.append(meanError)
				newWeights = doGradientDescent(meanError) # TODO

				notConverged = newWeights != self.weights

	def propagate(self, input):

		#pass all the inputs to the k hidden nodes		
		hiddenLayerOutput = map(lambda c: guassian(input, c, self.sigma), self.centers)

		result = sum(map(lambda x: x[0] * x[1], zip(hiddenLayerOutput, weights)))

		return result

def distance(x, y):
	return numpy.linalg.norm(x - y)

#the activation function of the hidden nodes
def guassian(input, center, sigma):
	#apply this function to the weighted sum at each hidden node
	#it will be the hyperbolic tangent function in our case
	distance = distance(input - center)

	return math.exp(-1 * ((distance**2) / (2 * sigma**2)))

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
