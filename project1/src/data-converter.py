#!/bin/python3
#!/bin/python

import argparse
import os
import inspect
from arff import ARFF

fileLocationsAndAttributes = {

    #ecoli
    '../data/ecoli/ecoli.data': {
        'relation': 'ecoli',
        'attributeNames': ['Sequence name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2'],
        'classNames': ['cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'],
    },

    #fertility
    '../data/fertility/fertility_Diagnosis.txt': {
        'relation': 'fertility',
        'attributeNames': ['Season', 'Age 18-36', 'Childish disease', 'Access/trauma', 'Surgery', 'Fevers', 'Alcohol consumption', 'Smoker', 'Hours spent sitting'],
        'classNames': ['N', 'O']
    },

    #glass
    '../data/glass/glass.data': {
        'relation': 'glass',
        'attributeNames': ['Id number', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'],
        'classNames': map(str, range(1, 8)), #'1'...'7'
    },

    #ionosphere
    '../data/ionosphere/ionosphere.data': {
        'relation': 'ionosphere',
        'attributeNames': map(str, range(1, 35)), #1..34 + class attr
        'classNames': ['g', 'b']
    },

    #magic
    '../data/magic/magic04.data': {
        'relation': 'magic04',
        'attributeNames': ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist'],
        'classNames': ['g', 'h']
    }
}

thisFileLocation = os.path.dirname(inspect.stack()[0][1])
def fileRelativeToHere(relativePath):
    return os.path.abspath(os.path.join(thisFileLocation, relativePath))

def main():
    outputDir = fileRelativeToHere('../output')

    for k in fileLocationsAndAttributes.keys():
        absolutePath = fileRelativeToHere(k)
        print('** ARFF-ing', absolutePath)
        print('** With features', fileLocationsAndAttributes[k]['attributeNames'])
        print('** And classes', fileLocationsAndAttributes[k]['classNames'])
        arff = ARFF(absolutePath, outputDir, fileLocationsAndAttributes[k])
        arff.printARFF()
        print('->', absolutePath, '\n')


if __name__ == '__main__': main()
