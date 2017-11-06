import os
import inspect
import numpy

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
