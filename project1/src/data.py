import re

whiteSpace = re.compile(r'\s+')

def processRows(inputListOfStrings, rowTransform):
    # take in list of strings:

    unjoined = None

    # if list is whitespace-separated...
    if len(inputListOfStrings[0].split(',')) == 1:
        unjoined = map(lambda s: whiteSpace.split(s.strip()), inputListOfStrings)

    # otherwise comma separated
    else:
        unjoined = map(lambda s: s.strip().split(','), inputListOfStrings)

    # unjoined is now a list of lists that represens the rows/columns of data
    # if there is a rowTransform do it othewise just join each row to a list
    if rowTransform is None:
        return map(lambda row: ",".join(row) + '\n', unjoined)
    else:
        return map(lambda row: ",".join(rowTransform(row)) + '\n', unjoined)
