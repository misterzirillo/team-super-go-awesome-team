from header import Header
from data import Data
from os import path

class ARFF:

    def __init__(self, fileLocation, attributeNames, outputDir):
        self.fileLocation = fileLocation
        self.attributeNames = attributeNames
        self.outputLocation = path.join(outputDir, path.splitext(path.basename(fileLocation))[0] + '.arff')
        self.header = Header(attributeNames)

        with open(fileLocation, 'r') as file:
            self.data = Data(file.readlines())

    def printARFF():
        pass
