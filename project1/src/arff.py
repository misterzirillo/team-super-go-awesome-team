from data import processRows
from os import path, makedirs

class ARFF:

    def __init__(self, fileLocation, outputDir, headerDetails):
        self.fileLocation = fileLocation
        self.headerDetails = headerDetails
        self.outputLocation = path.join(outputDir, path.splitext(path.basename(fileLocation))[0] + '.arff')

        rowTransform = headerDetails['rowFn'] if 'rowFn' in headerDetails else None
        with open(fileLocation, 'r') as file:
            self.dataLines = processRows(file.readlines(), rowTransform)

    def printARFF(self):

        if not path.exists(path.dirname(self.outputLocation)):
            try:
                makedirs(path.dirname(self.outputLocation))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(self.outputLocation, 'w') as file:

            #write header
            file.write('@Relation ' + self.headerDetails['relation'] + '\n')

            for attr in self.headerDetails['attributeNames']:
                file.write('@Attribute ' + attr + ' NUMERIC' + '\n')

            file.write('@Attribute class {' + ','.join(self.headerDetails['classNames']) + '}' + '\n')

            #write data
            file.write('@Data' + '\n')
            file.writelines(self.dataLines)
