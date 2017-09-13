import argparse
import os
import inspect
from arff import ARFF

fileLocationsAndAttributes = {
    '../data/ecoli/ecoli.data': ['a', 'b', 'c'],
    '../data/fertility/fertility_Diagnosis.txt': [],
    '../data/glass/glass.data': [],
    '../data/ionosphere/ionosphere.data': [],
    '../data/magic/magic04.data': []
}

thisFileLocation = os.path.dirname(inspect.stack()[0][1])
def fileRelativeToHere(relativePath):
    return os.path.abspath(os.path.join(thisFileLocation, relativePath))

def main():
    outputDir = fileRelativeToHere('../output')

    for k in fileLocationsAndAttributes.keys():
        absolutePath = fileRelativeToHere(k)
        print('ARFF-ing', absolutePath)
        print('With features', fileLocationsAndAttributes[k])
        arff = ARFF(fileLocation, attributeNames, outputDir)


if __name__ == '__main__': main()
