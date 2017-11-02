import os
import inspect
import numpy

# helper to write files with relative locations
thisFileLocation = os.path.dirname(inspect.stack()[0][1])


def fileRelativeToHere(relativePath):
    return os.path.abspath(os.path.join(thisFileLocation, relativePath))


# parse in a dataset from the data folder by name
# data should be in CSV form with the class as the last element on each line
def readDatasetFromFile(filename):
    data = []
    with open('../data/' + filename, 'r') as f:
        for line in f:
            linearr = list(map(float, line.split(',')))
            arg = numpy.asarray(linearr[:len(linearr) - 1])
            out = linearr[len(linearr) - 1]
            data.append((arg, out))
    return data
