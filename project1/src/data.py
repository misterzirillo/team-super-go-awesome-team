# I've tried to include a general idea of how this might work

class Data:

    # python's constructor syntax
    def __init__(self, inputListOfStrings):
        # inputListOfStrings is the lines of the input file
        self.rawInput = inputListOfStrings
        self.formatData()

    def formatData():
        # iterate over self.rawInput
        # self.formattedInput = self.rowInput.map(it => it.format()) ???
        self.formattedInput = self.rawInput.split()
        if len(self.formattedInput) == 1:
            self.formattedInput = self.rawInput.split(",")

        self.formatted.join(self.formatted, ",")

    def printData(dataName):
        # print to <dataName>-data.arff
        <dataName>-data.arff.write(self.formattedInput)

