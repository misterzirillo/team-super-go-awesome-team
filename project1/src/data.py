# I've tried to include a general idea of how this might work

class Data:

    # python's constructor syntax
    def __init__(self, inputListOfStrings):
        # inputListOfStrings is the lines of the input file
        self.dataStrings = inputListOfStrings
        self.formatData()

    def formatData(self):
        # iterate over self.rawInput
        # split each string in list
        # join with comma's
        if len(self.dataStrings[0].split()) == 1: #must be comma separated if split does nothing
            for s in self.dataStrings:
                s = ",".join(s.split(",")) #maybe should delete this... and just pass... but at least now our converter makes the computer do something
        else: #must be space separated if not comma separated
            for s in self.dataStrings:
                s = ",".join(s.split())
