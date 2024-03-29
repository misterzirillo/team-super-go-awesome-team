import argparse
import math
from datetime import datetime
import numpy as np


class MLPNetwork:

	def __init__(self, shape):

		# Network Basic Properties
		#self.transfer = transfer
		self.shape = shape
		self.layers = len(self.shape) - 1
		self.transfer = transfer

		# Data from Feed Forward
		# Need to save for backprop
		self.layer_in = []
		self.layer_out = []
		self.previous_delta = []
		self.weights = []

		# Weight Arrays
		for (i, j) in zip(self.shape[:-1], self.shape[1:]):
			self.weights.append(np.random.normal(scale=1, size=(j, i + 1)))
			self.previous_delta.append(np.zeros((j, i + 1)))
		
	# Runs data through network
	# data should be a x by n numpy array where n is the number of features
	def propagate(self, data):

		input_shape = data.shape[0]

		# Clear past data
		self.layer_in = []
		self.layer_out = []

		# Start Feed Forward
		for i in range(self.layers):
			# Base Layer Inputs
			if i == 0:
				layer_in = self.weights[0].dot(np.vstack([data.T, np.ones([1, input_shape])]))
			# Higher Layers
			else:
				layer_in = self.weights[i].dot(np.vstack([self.layer_out[-1], np.ones([1, input_shape])]))

			self.layer_in.append(layer_in)
			self.layer_out.append(self.transfer(layer_in, i == self.layers - 1))

		return self.layer_out[-1].T
	
def transfer(x, output):
	if not output:
		return np.tanh(x) # hidden layers will use hyperbolic tangent
	else:
		return 1 / (1 + np.exp(-x)) # output layer should squash to [0, 1]
