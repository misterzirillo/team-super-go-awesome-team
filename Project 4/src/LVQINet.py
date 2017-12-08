'''
Learning Vector Quantization
'''

import numpy as np
import helpers

class LVQI:

    eta = 0
    radius = 0
    shape = []
    weights = []

    def __init__(self, num_attrs, num_clusters):
        self.shape = (num_attrs, num_clusters)

    def weight_update(self):
        pass

    # returns the set of weights that were closest to the input pattern
    # aka a cluster center
    def propagate(self, input_pattern):
        return min(enumerate(self.weights), key=lambda weights: helpers.distance(input_pattern[1], weights))

    def output(self, training_data):
        # expecting training_data as a dictionary so need to extract keys
        np_data = np.fromiter(training_data, np.float64)

        self.train(np_data)

        for pattern in np_data:
            training_data[pattern] = self.propagate(pattern)

    # take some iterable input numpys
    # and learn the clusters
    def train(self, training_data):

        # copy this because we are annealing...
        working_eta = self.eta

        # get max/min for each attr
        maxes = np.nanmax(training_data, axis=0)
        mins = np.nanmin(training_data, axis=0)

        # initialize weights to random values inside the training data space
        self.weights = list(np.random.rand(*self.shape) * (maxes - mins) + mins)

        while True:  # TODO make stopping conditions
            for pattern in training_data:

                # find the closest weights
                winner_index, winner_weights = self.propagate(pattern)

                # update the weights
                winner_delta = working_eta * (pattern - winner_weights)
                self.weights[winner_index] = winner_weights + winner_delta
