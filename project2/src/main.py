import argparse
import numpy
from functools import reduce
import os
import inspect

# implementation of rosenbrock
def rosenbrock(*vals):

    if len(vals) < 2: raise ValueError('2 or more values required')

    def iteration(i):
        xCurrent = vals[i]
        xNext = vals[i + 1]
        return 100 * (xNext - xCurrent**2)**2 + (1 - xCurrent)**2

    return sum([iteration(i) for i in range(len(vals) - 1)])


# takes a set of argument vectors and maps them to a set of tuples
# each argument vector is paired with the rosenbrock output for that vector
def buildDataset(arguments):
    return list(map(lambda x: (x, rosenbrock(*x)), arguments))


# this function will generate a set of arguments with the given parameters
# argumentDimension: number of dimensions to use for argument vector > 1
# datasetSize: how many values in the dataset
def generateRandomArguments(argumentDimension, datasetSize):
    if argumentDimension > 1:
        return [numpy.random.uniform(-3., 3., argumentDimension) for i in range(datasetSize)]
    else:
        raise ValueError('argumentDimension < 2')


# generates a dataset with the given parameters and writes it in the current directory
def generateAndWriteDataset(argumentDimension, datasetSize, filename):
    with open(filename, 'w') as f:
        dataset = buildDataset(generateRandomArguments(argumentDimension, datasetSize))
        lines = map(lambda tup: ','.join(map(str, numpy.append(tup[0], tup[1]))) + '\n', dataset)
        f.writelines(lines)


# helper to write files with relative locations
thisFileLocation = os.path.dirname(inspect.stack()[0][1])
def fileRelativeToHere(relativePath):
    return os.path.abspath(os.path.join(thisFileLocation, relativePath))


# parse in a dataset from the data folder by name
def readDatasetFromFile(filename):
    data = []
    with open('../data/' + filename, 'r') as f:
        for line in f:
            linearr = list(map(float, line.split(',')))
            arg = numpy.asarray(linearr[:len(linearr) - 2])
            out = linearr[len(linearr) - 1]
            data.append( (arg, out) )
    return data


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
if __name__ == '__main__':
    # make parser
    parser = argparse.ArgumentParser()
    parser.add_argument('networkType', choices=['mlp', 'rbf'], help='The type of network')
    parser.add_argument('dimension', type=int, help='The number of dimensions to use')
    parser.add_argument('layerConfig', nargs='*', type=int, help='The number of nodes in each hidden layer. Each value represents a layer.')
    args = parser.parse_args()
    makeNetwork(args.networkType, args.dimension, args.layerConfig)