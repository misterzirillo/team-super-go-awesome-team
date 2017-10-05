import argparse
import numpy
from functools import reduce

# implementation of rosenbrock
def rosenbrock(*vals):

    if len(vals) < 2: raise ValueError('2 or more values required')

    def iteration(i):
        xCurrent = vals[i]
        xNext = vals[i + 1]
        return 100 * (xNext - xCurrent**2)**2 + (1 - xCurrent)**2

    return sum([iteration(i) for i in range(len(vals) - 1)])

# this function will build a dataset with the given parameters
# argumentDimension: number of dimensions to use for argument vector > 1
# datasetSize: how many values in the dataset
# returns a dictionary where the key is the argument tuple and the value is the computed rosenbrock
# argument values are restricted -10..10. not sure if this is proper but we need some boundaries
def buildDataset(argumentDimension, datasetSize):
    ret = dict()
    for i in range(size):
        rosenArgs = tuple(numpy.random.uniform(-10., 10., dimension))
        rosenVal = rosenbrock(*rosenArgs)
        ret[rosenArgs] = rosenVal
    return ret

# this fn is the meat and potatos of the driver
# it validates input and orchestrates the creation and training FOR ROSENBROCK
# networkType: either mlp or rbf
# dimension: the rosenbrock dimension
def constructNetwork(networkType, dimension, layerConfig=[]):
    
    # validate input
    if (networkType != 'mlp' and networkType != 'rbf'):
        raise ValueError('networkType must be \'mlp\' or \'rbf\'')

    if (dimension is not int or dimension < 1):
        raise ValueError('dimension must be int > 1')

    # create & train
    if (networkType == 'mlp'):

        #mlp must have layerconfig
        if (len(layerConfig) < 1): 
            raise ValueError('MLP must supply layerConfig')
        elif (not all(isinstance(e, int) and e > 0 for e in layerConfig)):
            raise ValueError('MLP layerConfig values must be int > 0')

        #create mlp
        #train mlp
        #return mlp
        pass
    else:
        #create rbf
        #train rbf
        #return rbf
        pass




# when executing as script...

# make parser
parser = argparse.ArgumentParser()
parser.add_argument('networkType', choices=['mlp', 'rbf'], help='The type of network')
parser.add_argument('dimension', type=int, help='The number of dimensions to use')
parser.add_argument('layerConfig', nargs='*', type=int, help='The number of nodes in each hidden layer. Each value represents a layer.')

if __name__ == '__main__':
    args = parser.parse_args()
    makeNetwork(args.networkType, args.dimension, args.layerConfig)