
'''
Layer is really just a holder for other data.
If anyone comes up with some greater plan for this
class go ahead and implement.
'''
class Layer:

    __init__(self, nodes):
        self.nodes = nodes
        self.pLayer = None
        self.sLayer = None