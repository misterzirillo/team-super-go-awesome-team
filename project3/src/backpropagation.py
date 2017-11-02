import network

class Backpropagation(object):

	def __init__(self, shape, transfer, transfer_derivative, error_function):
		self.network = network.MLPNetwork(shape, transfer)
		self.shape = self.network.shape
		self.transfer_derivative = transfer_derivative
		self.error_function = error_function

	# same as propagate, but then backpropagates the
	# error using gradient descent
	def backpropagate(self, training_data, batch_size, timelimit_minutes = 60, convergence_threshold = .01, max_epoch = 100000):
        
		batches = [training_data]
		printcount = 0
		batch_mse = []
		epoch = 0
		start = datetime.now()

        def time_ok():
            if (datetime.now() - start).total_seconds() / 60  > timelimit_minutes:
                print('Backpropogation aborted - time limit reached')
                return False
            else:
                return True

        def converged():
            converged = batch_mse[-1] < convergence_threshold
            if converged:
                print('Backpropogation successful - convergence threshold reached')
            return converged

        def diverged():
            diverged = math.isnan(batch_mse[-1])
            if diverged:
                print('Backpropagation aborted - diverged')
            return diverged

		def epochs_exceeded():
			exceeded = epoch > max_epoch
			if (exceeded):
				print('Backpropagation aborted - max epochs exceeded')       
			return exceeded

        while not converged() and time_ok() and not diverged() and not epochs_exceeded():

            # return batches
            for batchnum, batch in enumerate(batches):

                delta = []
                batch_examples = x.shape[0]
                self.network.propagate(x)

                for i in reversed(range(self.network.layers)):
                    if i == self.network.layers - 1:
                        out_delta = self.network.layer_out[i] - y.T
                        error = np.mean(out_delta ** 2)  # MSE
                        d = out_delta / len(y)
                        delta.append(d)

                    else:
                        int_delta = self.network.weights[i + 1].T.dot(delta[-1])
                        delta.append(int_delta[:-1, :] * self.transfer_derivative(self.network.layer_in[i], False))

                for i in range(self.network.layers):
                    delta_i = self.network.layers - 1 - i

                    if i == 0:
                        layer_out = np.vstack([x.T, np.ones([1, batch_examples])])
                    else:
                        layer_out = np.vstack([self.network.layer_out[i - 1], np.ones([1, self.network.layer_out[i - 1].shape[1]])])

                    cur_weight_delta = np.sum( \
                        layer_out[None, :, :].transpose(2, 0, 1) * \
                        delta[delta_i][None, :, :].transpose(2, 1, 0), axis=0)

                    weight_delta = self.network.learning_rate * cur_weight_delta + self.network.momentum * self.network.previous_delta[i]

                    self.network.weights[i] -= weight_delta

                    self.network.previous_delta[i] = weight_delta

                batch_mse.append(error)
                printcount += batch_examples
                if printcount > 10000:
                    s = 'epoch\t{}\tbatch\t{}\terror_mse\t{}'.format(epoch, batchnum, error)
                    print(s)
                    printcount = 0

            epoch += 1

        return batch_mse

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

def transfer(x, output):
	pass