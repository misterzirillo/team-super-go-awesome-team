import re

whiteSpace = re.compile('\s*')

def processRows(inputListOfStrings, rowTransform):
    # take in list of strings:
    # if list is whitespace-separated...

    unjoined = None
    if len(inputListOfStrings[0].split(',')) == 1:
        unjoined = map(lambda s: whiteSpace.split(s.strip()), inputListOfStrings)

    # otherwise comma separated
    else:
        unjoined = map(lambda s: s.strip().split(','), inputListOfStrings)

    if rowTransform is None:
        return map(lambda row: ",".join(row) + '\n', unjoined)
    else:
        return map(lambda row: ",".join(rowTransform(row)) + '\n', unjoined)
