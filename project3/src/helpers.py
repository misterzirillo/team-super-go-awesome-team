import os
import inspect
import numpy
from itertools import takewhile
import random

# helper to write files with relative locations
thisFileLocation = os.path.dirname(inspect.stack()[0][1])


def fileRelativeToHere(relativePath):
    return os.path.abspath(os.path.join(thisFileLocation, relativePath))


# parse in a dataset from the data folder by name
# argStart index is where to start slicing a line to capture all args
# argEnd index is were to stop slicing args
# class is the index where the class label is located
# example for zoo dataset
# x, y = helpers.readDatasetFromFile('zoo.data', 1, -1, -1)
# zoo dataset has a non-feature in the first position so start looking at 
# index 1. features continue until the last index, which is the class label
# output is a tuple (x, y) where x is a matrix of input vectors, y is a matrix of output vectors
def readDatasetFromFile(filename, argStartIndex, argEndIndex, classIndex):
    x = []
    y = []
    outs = {}
    outCount = 0
    with open('../data/' + filename, 'r') as f:
        for line in f:
            linearr = line.split(',')
            arg = numpy.fromiter(map(float, linearr[argStartIndex : argEndIndex]), numpy.float64)
            
            output = linearr[-1]
            if output not in outs:
                outs[output] = outCount
                outCount += 1

            x.append(arg)
            y.append(output)
            
    possibleYs = len(outs)
    ytransformed = []
    for it in y:
        zeros = [0] * possibleYs
        zeros[outs[it]] = 1
        ytransformed.append(numpy.array(zeros))

    return (numpy.array(x), numpy.array(ytransformed))

def readZooData():
    return readDatasetFromFile('zoo.data', 1, 7, -1)

def readLeafData():
    return readDatasetFromFile('leaf.csv', 2, 16, 0)

# CV helpers
def get2Fold(x, y):
    zipped = list(zip(x,y))
    return get_mini_batches(zipped, int(len(x) / 2) + 1)
    
# given a dataset, return minibatches using a random sampling with replacement mechanism
def get_mini_batches(dataset, batch_size):
	random_idxs = numpy.random.choice(len(dataset), len(dataset), replace=False)
	X_shuffled = list(map(lambda i: dataset[i], random_idxs))
	mini_batches = [X_shuffled[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
	return mini_batches

def rankBasedSelection(pop, sortFit, numParents):
    parents = []
    n = len(pop)
    #expected number of offspring generated by the best individual
    lam1 = 1.99
    #expected number of offspring generated by the best individual
    lam2 = 2-lam1

    Pxi = []
    wheel = []

    #assign probability of selection
    #calculate cumulative fitness and make roulette wheel
    for i in range(len(pop)):
        #normalizer
        Pxi.append((lam2 + (i/(n-1)) * (lam1 - lam2)) / n)
        wheel.append(sum(Pxi))

    parents = random.choices(sortFit, cum_weights=wheel, k=numParents)

    ''' while len(parents)< numParents:
        num = random.uniform(0, max(wheel, key = lambda x: x[0])[0])

        if len(wheel) is not 0:
            chosenOne = list(takewhile(lambda g: g[0] < num, wheel))[-1]
        else:
            print(wheel)
            raise "More parents than population"
        
        parents.append(chosenOne[1])
        del wheel[wheel.index(chosenOne)]
 '''
    return parents