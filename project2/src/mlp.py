import numpy as np


class MLPNetwork(object):

    layers = 0
    shape = None
    weights = []


    def __init__(self, inputs, outputs, transfer, learning_rate, momentum, hidden=[]):

        # Network Basic Properties
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.function = transfer
        self.c = 1.0

        # Layer Properties
        self.shape = [inputs]
        for i in range(len(hidden)):
            self.shape.append(hidden[i])
        self.shape.append(outputs)
        self.layers = len(self.shape) - 1

        # Data from Feed Forward
        self.layer_in = []
        self.layer_out = []
        self.previous_delta = []

        # Weight Arrays
        for (i, j) in zip(self.shape[:-1], self.shape[1:]):
            self.weights.append(np.random.normal(scale=0.1, size=(j, i + 1)))
            self.previous_delta.append(np.zeros((j, i + 1)))

    # Runs data through network
    def feed_forward(self, train_data):
        inputs = self.shape[0]

        # Clear past data
        self.layer_in = []
        self.layer_out = []

        # Start Feed Forward
        for i in range(self.layers):
            # Base Layer Inputs
            if i == 0:
                layer_in = self.weights[0].dot(np.vstack([train_data.T, np.ones([1, inputs])]))
            # Higher Layers
            else:
                layer_in = self.weights[i].dot(np.vstack([self.layer_out[-1], np.ones([1, inputs])]))

            self.layer_in.append(layer_in)
            self.layer_out.append(self.transfer(layer_in))

        return self.layer_out[-1].T

    def train(self, x, y):

        delta = []
        inputs = x.shape[0]

        self.feed_forward(x)

        for i in reversed(range(self.layers)):
            if i == self.layers - 1:
                out_delta = self.layer_out[i] - y.T
                error = np.sum(out_delta**2)
                delta.append(out_delta * self.d_transfer(self.layer_in[i]))
            else:
                int_delta = self.weights[i + 1].T.dot(delta[-1])
                delta.append(int_delta[:-1, :] * self.d_transfer(self.layer_in[i]))

        for i in range(self.layers):
            delta_i = self.layers - 1 - i

            if i == 0:
                layer_out = np.vstack([x.T, np.ones([1, inputs])])
            else:
                layer_out = np.vstack([self.layer_out[i - 1], np.ones([1, self.layer_out[i - 1].shape[1]])])

            cur_weight_delta = np.sum(\
                                  layer_out[None,:,:].transpose(2, 0, 1) *\
                                  delta[delta_i][None, :, :].transpose(2, 1, 0), axis = 0)

            weight_delta = self.learning_rate * cur_weight_delta + self.momentum * self.previous_delta[i]

            self.weights[i] -= weight_delta

            self.previous_delta[i] = weight_delta

        return error

    def transfer(self, x):
        if self.function == 'sig':
            return 1 / (1+np.exp(-x))
        elif self.function == 'lin':
            return self.c * x

    def d_transfer(self, x):
        if self.function == 'sig':
            temp = self.transfer(x)
            return temp * (1 - temp)
        elif self.function == 'lin':
            return self.c

if __name__ == "__main__":
    MLP = MLPNetwork(2, 1, 'sig', 0.1, 0.9, [3, 2])
    print(MLP.shape)
    print(MLP.weights)
