import argparse

# implementation of rosenbrock
def rosenbrock(*args):
    def iteration(i):
        xCurrent = args[i]
        xNext = args[i + 1]
        return 100 * (xNext - xCurrent**2)**2 + (1 - xCurrent)**2

    return sum([iteration(i) for i in range(len(args) - 1)])

# this fn is the meat and potatos of the driver
# it validates input and orchestrates the creation and training
# of either an mlp or rbf network
def makeNetwork(networkType, dimension, layerConfig):
    
    # validate input
    if (networkType != 'mlp' and networkType != 'rbf'):
        print('Invalid network type: ' + networkType)
        return

    if (dimension < 1):
        print('Invalid dimension < 1: ' + dimension)
        return

    # create & train
    if (networkType == 'mlp'):

        #mlp must have layerconfig
        if (len(layerConfig) < 1): 
            print('MLP network must supply layerConfig')
            return
        elif (not all(isinstance(e, int) and e > 0 for e in layerConfig)):
            print('Invalid layerConfig; must provide integers > 0: ' + str(layerConfig))
            return

        #create mlp
        #train mlp
        #return mlp
        pass
    else:
        #create rbf
        #train rbf
        #return rbf
        pass




# when executing as script...

# make parser
parser = argparse.ArgumentParser()
parser.add_argument('networkType', choices=['mlp', 'rbf'], help='The type of network')
parser.add_argument('dimension', type=int, help='The number of dimensions to use')
parser.add_argument('layerConfig', nargs='*', type=int, help='The number of nodes in each hidden layer. Each value represents a layer.')

if __name__ == '__main__':
    args = parser.parse_args()
    makeNetwork(args.networkType, args.dimension, args.layerConfig)