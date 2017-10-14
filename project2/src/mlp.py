import random


class MLPNetwork(object):
    layers = []
    learning_rate = 0.1
    momentum = 0.9      # Set as constant for now

    # Creates the Feedforward Network
    def __init__(self, n_inputs, n_outputs, nodes_per_layer=[]):
        self.layers.append(Layer(n_inputs, 0, self))
        for i in range(len(nodes_per_layer)):
            self.layers.append(Layer(nodes_per_layer[i], i+1, self))
        self.layers.append(Layer(n_outputs, len(nodes_per_layer)+1, self))

    def feed_forward(self, data_in=[]):
        for layer in range(len(self.layers)):
            if layer == 0:
                self.layers[layer].activate_base_layer(data_in)
                pass
            else:
                self.layers[layer].activate_layer()


class Layer(object):
    nodes = []
    layer_number = 0
    n_nodes = 0
    parent = []

    def __init__(self, n_nodes, layer_number, in_parent):
        self.n_nodes = n_nodes
        self.layer_number = layer_number
        self.parent = in_parent
        self.nodes = []
        if layer_number == 0:
            for i in range(n_nodes):
                self.nodes.append(Node(1, self))
            pass
        else:
            for i in range(n_nodes):
                self.nodes.append(Node(in_parent.layers[layer_number-1].n_nodes, self))
            pass

    def activate_layer(self):
        for i in range(self.n_nodes):
            self.nodes[i].activate_node()

    def activate_base_layer(self, data_in=[]):
        for i in range(len(data_in)):
            self.nodes[i].activate_base_node(data_in[i])


class Node(object):

    weights = []
    prev_change = []
    sum = 0
    transfer = 0
    parent = []

    def __init__(self, n_inputs, in_parent):
        self.n_inputs = n_inputs
        self.weights = [random.random() for i in range(self.n_inputs + 1)]
        self.parent = in_parent

    def update_weights(self, connection, weight):
        self.weights[connection] = weight

    # Just passes sum at this point, need to change
    def transfer_function(self, in_sum):
        return in_sum

    # Activates the node and updates self.transfer
    def activate_node(self):
        t_sum = 0
        for j in range(len(self.weights) - 1):
            t_sum += self.weights[j] * self.parent.parent.layers[self.parent.layer_number - 1].nodes[j].transfer
        t_sum += self.weights[len(self.weights) - 1]
        self.transfer =  self.transfer_function(t_sum)

    # Base Layer passes input data through
    def activate_base_node(self, input_data):
        self.transfer = input_data


def rosenbrock(*vals):

    if len(vals) < 2: raise ValueError('2 or more values required')

    def iteration(i):
        xCurrent = vals[i]
        xNext = vals[i + 1]
        return 100 * (xNext - xCurrent**2)**2 + (1 - xCurrent)**2

    return sum([iteration(i) for i in range(len(vals) - 1)])


def buildDataset(argumentDimension, datasetSize):
    ret = dict()
    for i in range(datasetSize):
        rosenArgs = tuple(random.uniform(-10., 10., argumentDimension))
        rosenVal = rosenbrock(*rosenArgs)
        ret[rosenArgs] = rosenVal
    return ret


def constructNetwork(networkType, dimension, layerConfig=[]):
    # validate input
    if (networkType != 'mlp' and networkType != 'rbf'):
        raise ValueError('networkType must be \'mlp\' or \'rbf\'')

    if (dimension < 1):
        raise ValueError('dimension must be int > 1')

    #data = buildDataset(dimension, 100)
    data = [3,2]

    # create & train
    if (networkType == 'mlp'):

        # mlp must have layerconfig
        if (len(layerConfig) < 1):
            raise ValueError('MLP must supply layerConfig')
        elif (not all(isinstance(e, int) and e > 0 for e in layerConfig)):
            raise ValueError('MLP layerConfig values must be int > 0')

        TestNetwork = MLPNetwork(dimension, 1, layerConfig)
        TestNetwork.feed_forward(data)


        # create mlp
        # train mlp
        # return mlp
        pass
    else:
        # create rbf
        # train rbf
        # return rbf
        pass


constructNetwork('mlp', 2, [3, 2])