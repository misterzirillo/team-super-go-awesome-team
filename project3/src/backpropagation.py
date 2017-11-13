from mlpnetwork import MLPNetwork
from datetime import datetime
import math
import numpy as np
import helpers

class Backpropagation(object):

	def __init__(self, shape, learning_rate = 0.01, momentum = 0.5):
		self.network = MLPNetwork(shape)
		self.shape = self.network.shape
		self.transfer_derivative = lambda x, isOutput: x * (1 - x) if isOutput else np.cosh(x) ** -2
		self.learning_rate = learning_rate
		self.momentum = momentum

	# same as propagate, but then backpropagates the
	# error using gradient descent
	def train(self, x, y, validationX, validationY, maxEpoch):

		self.trainingErrors = []
		self.validationErrors = []
		
		printcount = 0
		epoch = 0
		start = datetime.now()
		converged = False

		def epochs_exceeded():
			exceeded = epoch > maxEpoch
			if (exceeded):
				print('Backpropagation aborted - max epochs exceeded')       
			return exceeded

		while not converged and not epochs_exceeded():

			delta = []
			batch_examples = x.shape[0]
			self.network.propagate(x)

			for i in reversed(range(self.network.layers)):
				if i == self.network.layers - 1:
					netout = self.network.layer_out[i]
					out_delta = (netout - y.T)
					error = helpers.percentCorrect(netout.T, y)
					d = out_delta
					delta.append(d)

				else:
					int_delta = self.network.weights[i + 1].T.dot(delta[-1])
					delta.append(int_delta[:-1, :] * self.transfer_derivative(self.network.layer_in[i], False))

			for i in range(self.network.layers):
				delta_i = self.network.layers - 1 - i

				if i == 0:
					layer_out = np.vstack([x.T, np.ones([1, batch_examples])])
				else:
					layer_out = np.vstack([self.network.layer_out[i - 1], np.ones([1, self.network.layer_out[i - 1].shape[1]])])

				cur_weight_delta = np.sum( \
					layer_out[None, :, :].transpose(2, 0, 1) * \
					delta[delta_i][None, :, :].transpose(2, 1, 0), axis=0)

				weight_delta = self.learning_rate * cur_weight_delta + self.momentum * self.network.previous_delta[i]

				self.network.weights[i] -= weight_delta

				self.network.previous_delta[i] = weight_delta

			printcount += batch_examples
			valSuccess = helpers.percentCorrect(self.network.propagate(validationX), validationY)
			if printcount > 10000:
				s = 'epoch {}\ttraining percent correct {}\tvalidation percent correct {}'.format(epoch, error,valSuccess)
				print(s)
				printcount = 0

			epoch += 1
			converged = self.postIterationProcess(validationX, validationY, error, epoch, valSuccess)

	# test against the validation set and record results
	# perform convergence check
	def postIterationProcess(self, x, y, fit, t, valSuccess):
		trainingError = 100 - fit
		validationError = 100 - valSuccess

		#print('training error: ' + str(trainingError) + '\tvalidation error: ' + str(validationError))

		self.trainingErrors.append(trainingError)
		self.validationErrors.append(validationError)
		converged = False
		#converged = self.validationErrors[-1] > self.validationErrors[-2] if len(self.validationErrors) > 10 else False
		if t > 100:
			converged = self.validationErrors[-1] > np.mean(self.validationErrors) + np.std(self.validationErrors)
			if converged:
				print("Convergence check reached at Generation " + str(t))
				print(str(self.validationErrors[-1]) + ' > ' + str(np.mean(self.validationErrors)) + str(np.std(self.validationErrors)))

		return converged