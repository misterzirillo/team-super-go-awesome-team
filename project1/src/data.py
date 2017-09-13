# I've tried to include a general idea of how this might work

class Data:

    # python's constructor syntax
    def __init__(self, inputListOfStrings):
        # inputListOfStrings is the lines of the input file
        self.dataStrings = inputListOfStrings

    def formatData():
        # iterate over self.rawInput
        # self.formattedInput = self.rowInput.map(it => it.format()) ???
        for s in self.rawInput
            s = s.split()
            if len(s) == 1:
                s = s.split(",")
            ",".join(s)
        return self.dataStrings
        pass

    def printData(dataName):
        # print to <dataName>-data.arff
        <dataName>-data.arff.write(self.dataStrings)
        pass