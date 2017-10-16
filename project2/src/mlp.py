import numpy as np
from rbf import get_mini_batches
from datetime import datetime
import math
import argparse
from main import readDatasetFromFile

class MLPNetwork(object):

    layers = 0
    shape = None
    weights = []


    def __init__(self, inputs, outputs, transfer, hidden=[]):

        # Network Basic Properties
        self.learning_rate = .1
        self.momentum = .5
        self.function = transfer
        self.c = 1.0

        # Layer Properties
        self.shape = [inputs]
        for i in range(len(hidden)):
            self.shape.append(hidden[i])
        self.shape.append(outputs)
        self.layers = len(self.shape) - 1

        # Data from Feed Forward
        self.layer_in = []
        self.layer_out = []
        self.previous_delta = []

        # Weight Arrays
        for (i, j) in zip(self.shape[:-1], self.shape[1:]):
            self.weights.append(np.random.normal(scale=0.1, size=(j, i + 1)))
            self.previous_delta.append(np.zeros((j, i + 1)))

    # Runs data through network
    def feed_forward(self, train_data):

        inputs = train_data.shape[0]

        # Clear past data
        self.layer_in = []
        self.layer_out = []

        # Start Feed Forward
        for i in range(self.layers):
            # Base Layer Inputs
            if i == 0:
                layer_in = self.weights[0].dot(np.vstack([train_data.T, np.ones([1, inputs])]))
            # Higher Layers
            else:
                layer_in = self.weights[i].dot(np.vstack([self.layer_out[-1], np.ones([1, inputs])]))

            self.layer_in.append(layer_in)
            self.layer_out.append(self.transfer(layer_in))

        return self.layer_out[-1].T

    def train(self, dataset):

        batchSize = 2 ** (self.shape[0] + 2)
        batches = get_mini_batches(dataset, batchSize)

        timelimitMinutes = 30
        batchMSEs =[]
        epoch = 0
        notConverged = True

        printcount = 0

        start = datetime.now()
        def checkTimeMinutes():
            return (datetime.now() - start).total_seconds() // 60

        while notConverged and checkTimeMinutes() < timelimitMinutes:

            #return batches
            for batchnum, batch in enumerate(batches):
                delta = []
                X, Y = zip(*batch)
                x = np.asarray(X, dtype='int64')
                y = np.asarray(Y)

                inputs = x.shape[0]

                self.feed_forward(x)

                for i in reversed(range(self.layers)):
                    if i == self.layers - 1:
                        out_delta = self.layer_out[i] - y.T
                        error = np.mean(out_delta**2)
                        delta.append(out_delta * self.d_transfer(self.layer_in[i]))
                    else:
                        int_delta = self.weights[i + 1].T.dot(delta[-1])
                        delta.append(int_delta[:-1, :] * self.d_transfer(self.layer_in[i]))

                for i in range(self.layers):
                    delta_i = self.layers - 1 - i

                    if i == 0:
                        layer_out = np.vstack([x.T, np.ones([1, inputs])])
                    else:
                        layer_out = np.vstack([self.layer_out[i - 1], np.ones([1, self.layer_out[i - 1].shape[1]])])

                    cur_weight_delta = np.sum(\
                                        layer_out[None,:,:].transpose(2, 0, 1) *\
                                        delta[delta_i][None, :, :].transpose(2, 1, 0), axis = 0)

                    weight_delta = self.learning_rate * cur_weight_delta + self.momentum * self.previous_delta[i]

                    self.weights[i] -= weight_delta

                    self.previous_delta[i] = weight_delta

                printcount += inputs
                if printcount > 1000:
                    batchMSEs.append(error)
                    print('epoch\t{}\tbatch\t{}\tmins_elapsed\t{}\terror_mse\t{}'
                                .format(epoch, batchnum, checkTimeMinutes(), error))
                    printcount = 0

            epoch += 1        

        return batchMSEs

    def transfer(self, x):
        if self.function == 'sig':
            return np.tanh(x)
        elif self.function == 'lin':
            return self.c * x

    def d_transfer(self, x):
        if self.function == 'sig':
            return np.cosh(x) ** -2
        elif self.function == 'lin':
            return self.c
    

if __name__ == "__main__":
    # make parser
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The name of the dataset in ../data to use')
    parser.add_argument('inputs', type=int, help='The number of input nodes (should match dataset)')
    parser.add_argument('layerConfig', nargs='*', type=int, help='The number of nodes in each hidden layer. Each value represents a layer.')
    args = parser.parse_args()

    if len(args.layerConfig) > 0 and not all(args.layerConfig):
        raise ValueError('Invalid layerconfig ' + layerConfig)

    data = readDatasetFromFile(args.dataset)

    network = MLPNetwork(args.inputs, 1, 'sig', args.layerConfig)
    error = network.train(data)

    netName = 'mlp-n{}'.format('-'.join(map(str, network.shape)))
    print('finished training network ' + netName)

    errorFile = netName + '-error.csv'
    print('printing MSE values to ' + errorFile)	
    with open(errorFile, 'w') as f:
        f.writelines('\n'.join(map(str, error)))

    summaryFile = netName + '-summary.txt'
    print('printing network summary to ' + summaryFile)
    with open(summaryFile, 'w') as f:
        f.write('n=' + str(args.inputs) + '\n')
        f.write('shape=' + str(network.shape) + '\n')
        f.write('weights=' + str(network.weights) + '\n')
