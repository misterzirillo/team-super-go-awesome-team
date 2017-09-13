# I've tried to include a general idea of how this might work

class Header:

    # python's constructor syntax
    def __init__(self, inputListOfStrings):
        # inputListOfStrings is the lines of the input file
        self.rawInput = inputListOfStrings
        self.formatHeader()

    def formatHeader():
        # iterate over self.rawInput
        # self.formattedInput = self.rowInput.map(it => it.format()) ???
        pass

    def printHeader(dataName):
        # print to <dataName>-header.arff
        pass
