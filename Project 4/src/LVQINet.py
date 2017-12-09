'''
Learning Vector Quantization
'''

import numpy as np
import helpers


class LVQI:

    def __init__(self, num_attrs, num_clusters, eta, beta, gamma):
        self.num_attrs, self.num_clusters = (num_attrs, num_clusters)
        self.eta = eta
        self.beta = beta
        self.gamma = gamma
        self.weights = None  # weights are set when train is called

    # returns the set of weights that were closest to the input pattern
    # aka a cluster center
    def propagate(self, input_pattern, guilt_factor=None):
        return min(
            enumerate(self.weights),
            key=lambda tup: helpers.distance(input_pattern, tup[1]) - (guilt_factor(tup[0]) if guilt_factor else 0))

    def output(self, training_data, max_epochs=1000, min_error=.1):

        # we dont want to mutate the original dictionary (may screw with other methods if dict is reused)
        # so copy it and return the copy after all is done
        new_clusters = training_data.copy()

        # expecting training_data as a dictionary so need to extract keys
        np_data = helpers.get_input_matrix_from_dict(training_data)

        # do clustering
        self.train(np_data, max_epochs, min_error)

        # change the dict value for each thing
        for pattern in np_data:
            key = tuple(pattern)
            new_clusters[key] = (new_clusters[key][0], self.propagate(pattern))

        return new_clusters

    # take some iterable input numpys
    # and learn the clusters
    def train(self, training_data, max_epochs, min_error):

        # initialize output node guilt for conscience factor
        # guilty nodes will win less even if they are the best
        # this keeps one cluster from dominating the others
        guilt = [0] * self.num_clusters
        winner_index = None

        def conscious_factor(i):
            if epoch == 0:
                return 1 / self.num_clusters
            else:
                new_guilt = guilt[i] + self.beta * (1 if winner_index == i else 0 - guilt[i])
                guilt[i] = new_guilt
                return self.gamma * (1 / self.num_attrs - new_guilt)

        # get max/min for each attr
        maxes = np.nanmax(training_data, axis=0)
        mins = np.nanmin(training_data, axis=0)

        # initialize weights to random values inside the training data space
        self.weights = list(np.random.rand(self.num_clusters, self.num_attrs) * (maxes - mins) + mins)

        epoch = 0
        keep_going = True
        quantization_error = 0
        while keep_going:

            epoch += 1
            error_sum = 0

            # iterate over all inputs
            for pattern in training_data:

                # find the closest weights "competition" without conscience
                winner_index, _ = self.propagate(pattern)

                # the no-guilt winner is found, so do it again with feelings
                winner_index, winner_weights = self.propagate(pattern, conscious_factor)

                # update the weights
                winner_delta = pattern - winner_weights
                learning_factor = self.eta * (1 - float(epoch) / max_epochs)
                self.weights[winner_index] = winner_weights + learning_factor * winner_delta

                # some record-keeping
                error_sum += np.sqrt(winner_delta.dot(winner_delta)) ** 2
                # print('Winning cluster ' + str(winner_index) + ' updated by ' + str(np.linalg.norm(winner_delta)))
                # print('Guilt ' + str(np.array(guilt)))

            quantization_error = error_sum / len(training_data)
            print(quantization_error)
            keep_going = quantization_error > min_error and epoch < max_epochs

        print('LVQI finished at epoch ' + str(epoch) + ' error ' + str(quantization_error))
