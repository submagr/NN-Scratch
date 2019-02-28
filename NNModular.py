import numpy as np
np.random.seed(123)


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-1.0*x))


def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))


def quadratic_cost_derivative(activation, y):
    return activation - y

def cross_entropy_derivative(a, y):
    return (y/a - (1-y)/(1-a))


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(s2, s1) for s1, s2 in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(s, 1) for s in self.sizes[1:]]

    def backprop(self, x, y):
        # feedforward
        activation = x
        activations = [activation]
        zs = []
        for l in xrange(self.num_layers-1):
            w, b = self.weights[l], self.biases[l]
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        # backprop
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta = cross_entropy_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, np.transpose(activations[-1 - 1]))

        for l in xrange(2, self.num_layers):
            delta = np.dot(np.transpose(self.weights[-l+1]), delta) * sigmoid_prime(zs[-l])
            nabla_w[-l] = np.dot(delta, np.transpose(activations[-l-1]))
            nabla_b[-l] = delta

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
        self.weights = [w - (eta/m)*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/m)*nb for b, nb in zip(self.biases, nabla_b)]

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
            if list(y).index(max(y)): # list(activation).index(max(activation)) == y: #
                correct_count += 1
            # else:
            #    print "expected {}, got {}".format(list(activation).index(max(activation)), y)
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
        # x2 = 2*x1 + 3 + error
        x2 = (x1+4)*(x1-2) + error
        y = np.transpose([[1, -1]] if error < 0 else [[-1, 1]])
        data.append((np.transpose([[x1, x2]]), y))
    return data

"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import cPickle
import gzip

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

if __name__ == "__main__":
    # training_data, validation_data, test_data = load_data_wrapper()
    train_size = 1000
    test_size = int(train_size*0.2)
    eta = 0.650
    sizes = [2, 3, 2]

    myNet = Network(sizes)

    data = gen_linear_data(train_size + test_size)

    training_data = [x for x in data[0:train_size]]
    test_data = [x for x in data[train_size: train_size + test_size]]

    myNet.sgd(training_data, 10, 50, eta, test_data)
