import numpy as np


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-1*x))


def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))


def quadratic_cost_derivative(activation, y):
    return activation - y


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.random((s2, s1)) for s1, s2 in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.random((s, 1)) for s in self.sizes[1:]]

    def backprop(self, x, y):
        # feedforward
        activation = x
        activations = [activation]
        zs = []
        for l in xrange(self.num_layers-1):
            w, b = self.weights[l], self.biases[l]
            z = np.dot(w, activation)
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        # backprop
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta = quadratic_cost_derivative(activation, y)
        nabla_b[-1] += delta
        nabla_w[-1] += np.dot(delta, np.transpose(activations[-1 - 1]))

        for l in xrange(2, self.num_layers):
            delta = np.dot(np.transpose(self.weights[-l+1]), delta) * sigmoid_prime(zs[-l])
            nabla_w[-l] += np.dot(delta, np.transpose(activations[-l-1]))
            nabla_b[-l] += delta

        return nabla_w, nabla_b

    def update_mini_batch(self, mini_batch, eta):
        # feedforward
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for datapoint in mini_batch:
            x, y = datapoint[0], datapoint[1]
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        m = len(mini_batch)
        self.weights = [w + (eta/m)*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b + (eta/m)*nb for b, nb in zip(self.biases, nabla_b)]

    def feedforward(self, x):
        activation = x
        for l in xrange(self.num_layers-1):
            z = np.dot(self.weights[l], activation) + self.biases[l]
            activation = sigmoid(z)
        return activation

    def evaluate(self, test_data):
        correct_count = 0
        for datapoint in test_data:
            x, y = datapoint[0], datapoint[1]
            activation = self.feedforward(x)
            if list(activation).index(max(activation)) == list(y).index(max(y)):
                correct_count += 1
        return (correct_count*1.0)/len(test_data)

    def sgd(self, training_data, n_epochs, batch_size, eta, test_data = None):
        n_test = 0
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for epoch in xrange(n_epochs):
            print "Training epoch ", epoch

            np.random.shuffle(training_data)
            mini_batches = [training_data[k: k + batch_size] for k in xrange(0, n, batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print "Epoch {0} trained : accuracy {1} among {2} test samples\n"\
                    .format(epoch, self.evaluate(test_data), len(test_data))
            else:
                print "Epoch {0} trained\n".format(epoch)


def gen_linear_data(size):
    data = []
    for i in xrange(size):
        error = np.random.uniform(-1, 1)
        x1 = np.random.uniform(-1, 1)
        x2 = 2*x1 + 3 + error
        # x2 = (x1+4)*(x1-2)
        y = np.transpose([[1, 0]])if error < 0 else np.transpose([[0, 1]])
        data.append((np.transpose([[x1, x2]]), y))
    return data


if __name__ == "__main__":
    train_size = 1000
    test_size = int(train_size*0.2)
    eta = 0.6
    sizes = [2, 4, 2]
    np.random.seed(123)

    myNet = Network(sizes)

    data = gen_linear_data(train_size + test_size)

    train_data = [x for x in data[0:train_size]]
    test_data = [x for x in data[train_size: train_size + test_size]]

    myNet.sgd(train_data, 10, 20, eta, test_data)
