#%% Imports

from __future__ import print_function

import gzip
import json
import os
import pickle

import numpy as np

import tensorflow as tf

from math import ceil

from operator import mul
from functools import reduce

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'


# %% Provided functions

def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes, ))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data', recs=None):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    if recs is not None:
        test_x, test_y = [arr[:recs] for arr in test_set]
        valid_x, valid_y = [arr[:recs] for arr in valid_set]
        train_x, train_y = [arr[:recs] for arr in train_set]
    else:
        test_x, test_y = test_set
        valid_x, valid_y = valid_set
        train_x, train_y = train_set

    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(
        valid_y), test_x, one_hot(test_y)


#%% Part A ###############################################

########### Extra functions


def read_from_json(name):
    fname = os.path.join(".", "results", name + ".json")

    if os.path.exists(fname):
        with open(fname, "r") as fh:
            results = json.load(fh)

    return results


def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def make_batches(x, y, batch_size):
    assert x.shape[0] == y.shape[0]
    for i in range(ceil(x.shape[0] / batch_size)):
        yield x[batch_size * i:batch_size *
                (i + 1), ...], y[batch_size * i:batch_size * (i + 1), ...]


def results_dump(path, name, lr, filter_size, num_filters, batch_size,
                 learning_curve):
    results = dict()
    results["name"] = name
    results["lr"] = lr
    results["filter_size"] = filter_size
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["learning_curve"] = learning_curve

    with open(os.path.join(path, name + ".json"), "w") as fh:
        json.dump(results, fh)


###########


class Kerosey(object):
    """
    my ghetto implementation of a usable API for tensorflow. enjoy.

    main limitations: 
        - uses the default tf Graph, so you can't usefully create more than
        one model at a time.

        - only the layer types needed for this exercise have been defined, but you can
        play with them / rearrange them as you like.
    """

    class Layer(object):
        def __init__(self):
            self.shape = None
            self.func = None

    class InputLayer(Layer):
        def __init__(self, x_shape):
            self.shape = (-1, ) + x_shape[1:]
            self.func = lambda x: x

    class MultiLayer(Layer):
        def __init__(self, last_layer, const):
            self.shape = last_layer.shape
            self.func = lambda x: const * x

    class Conv2D(Layer):
        def __init__(self, last_layer, filter_size, num_filters, stride):
            # depth of input gets replaced with n_f
            self.shape = last_layer.shape[:-1] + (num_filters, )
            self.weights = tf.get_variable(
                'w' + str(id(self)),
                shape=(filter_size, filter_size, last_layer.shape[-1],
                       num_filters),
                initializer=tf.contrib.layers.xavier_initializer())
            self.bias = tf.get_variable(
                'b' + str(id(self)),
                shape=(num_filters, ),
                initializer=tf.contrib.layers.xavier_initializer())

            self.func = lambda x: tf.nn.bias_add(
                tf.nn.conv2d(
                    x, self.weights, strides=[
                        1, stride, stride, 1], padding='SAME'), self.bias
            )

    class MaxPool(Layer):
        def __init__(self, last_layer, pool_size):
            self.shape = (last_layer.shape[0],
                          ceil(last_layer.shape[1] / pool_size),
                          ceil(last_layer.shape[2] / pool_size),
                          last_layer.shape[3])

            self.func = lambda x: tf.nn.max_pool(
                x, ksize=[1, pool_size, pool_size, 1],
                strides=[1, pool_size, pool_size, 1], padding='SAME')

    class Relu(Layer):
        def __init__(self, last_layer):
            self.shape = last_layer.shape

            self.func = lambda x: tf.nn.relu(x)

    class Flatten(Layer):
        def __init__(self, last_layer):
            # nultiply out dimensions except the first (batch) one
            self.shape = (last_layer.shape[0],
                          reduce(lambda x, y: x * y, last_layer.shape[1:], 1))

            self.func = lambda x: tf.reshape(x, self.shape)

    class Dense(Layer):
        def __init__(self, last_layer, num_units):
            # nultiply out dimensions except the first (batch) one
            self.shape = (last_layer.shape[0], num_units)
            self.weights = tf.get_variable(
                'w' + str(id(self)),
                shape=(last_layer.shape[-1], num_units),
                initializer=tf.contrib.layers.xavier_initializer())
            self.bias = tf.get_variable(
                'b' + str(id(self)),
                shape=(num_units, ),
                initializer=tf.contrib.layers.xavier_initializer())

            self.func = lambda x: tf.add(tf.matmul(x, self.weights), self.bias)

    class LayeredModel(object):
        def __init__(self):
            self.layers = []
            tf.reset_default_graph()  #seems like a good time to do this

        def setup_placeholders(self, x_data, y_data):
            self.layers = [Kerosey.InputLayer(x_data.shape)]

            self.tfp_x = tf.placeholder("float", (None, ) + x_data.shape[1:])
            self.tfp_y = tf.placeholder("float", (None, ) + y_data.shape[1:])

        def add_layer(self, layer_type, *args):
            self.layers.append(layer_type(self.layers[-1], *args))

        def compile(self, loss='crossentropy'):
            self.predictor = reduce(lambda f, g: lambda x: g(f(x)),
                                    [l.func for l in self.layers], lambda x: x)

            if loss == 'crossentropy':
                self.loss_fn = tf.losses.softmax_cross_entropy
            elif loss == 'mse':
                self.loss_fn = tf.losses.mean_squared_error
            else:
                raise("lost")
                
            self.session = tf.Session(config=config)
            self.session.run(tf.global_variables_initializer())

            self.train_losses = []
            self.train_accs = []
            self.valid_accs = []
            self.test_acc = None

        def prediction(self):
            return self.predictor(self.tfp_x)

        def accuracy(self):
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.argmax(self.tfp_y, 1),
                        tf.argmax(self.prediction(), 1)), tf.float32))

        def loss(self):
            tf_losses = self.loss_fn(self.tfp_y, self.prediction())
            return tf.reduce_mean(tf_losses)

        def crunch(self, x, y, batch_size, results, collect_stats=False):
            rr = None  #this will be a list

            for x_batch, y_batch in make_batches(x, y, batch_size):

                r = self.session.run(
                    results,
                    feed_dict={
                        self.tfp_x: x_batch,
                        self.tfp_y: y_batch
                    })

                if collect_stats:
                    if rr is None:
                        rr = [v * x_batch.shape[0] for v in r]
                    else:
                        rr = [
                            np.add(vv, v * x_batch.shape[0])
                            for v, vv in zip(r, rr)
                        ]

            if collect_stats:
                return [v / x.shape[0] for v in rr]
            else:
                pass

        def train(self, x_train, y_train, x_valid, y_valid, lr, epochs,
                  batch_size):

            optimiser = tf.train.GradientDescentOptimizer(
                learning_rate=lr).minimize(self.loss())

            for i in range(epochs):

                self.crunch(x_train, y_train, batch_size, optimiser)

                # all batches complete

                train_loss, train_acc = self.crunch(
                    x_train, y_train, 1024,
                    [self.loss(), self.accuracy()], True)

                valid_acc, = self.crunch(x_valid, y_valid, 1024, [
                    self.accuracy(),
                ], True)

                print("Iter " + str(i) + ", Loss= " +
                      "{:.4f}".format(train_loss) + ", Train. Acc= " +
                      "{:.4f}".format(train_acc) +
                      ", Val. Acc={:.4f}".format(valid_acc))

                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                self.valid_accs.append(valid_acc)

            print("Training finished")

        def test(self, x_test, y_test, batch_size=1024):
            self.test_acc = self.crunch(x_test, y_test, batch_size,
                                        [self.accuracy()], True)
            print("Test Accuracy {:4f}".format(self.test_acc))
            pass

        def save(self, directory):
            save_path = tf.train.Saver().save(
                self.session, os.path.join('.', directory, 'model.ckpt'))
            print("Model saved in path: %s" % save_path)
            pass

        def restore(self, directory):
            save_path = tf.train.Saver().restore(
                self.session, os.path.join('.', directory, 'model.ckpt'))
            print("Model restored from path: %s" % save_path)
            pass


# Useful since we're going to be using this architecture a lot


def create_the_model(x, y, filter_size, num_filters):

    krmodel = Kerosey.LayeredModel()
    krmodel.setup_placeholders(x, y)
    krmodel.add_layer(Kerosey.Conv2D, filter_size, num_filters, 1)
    krmodel.add_layer(Kerosey.Relu)
    krmodel.add_layer(Kerosey.MaxPool, 2)
    krmodel.add_layer(Kerosey.Conv2D, filter_size, num_filters, 1)
    krmodel.add_layer(Kerosey.Relu)
    krmodel.add_layer(Kerosey.MaxPool, 2)
    krmodel.add_layer(Kerosey.Flatten)
    krmodel.add_layer(Kerosey.Dense, 128)
    krmodel.add_layer(Kerosey.Relu)
    krmodel.add_layer(Kerosey.Dense, 10)
    krmodel.compile()

    return krmodel


if __name__ == '__main__':
    # %% Load Data

    # put this in ~/data so i'm not downloading the same thing into 100 different directories
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(
        os.path.expanduser(os.sep.join(["~", "data"])), recs=500)

    results_path = os.path.join(".", "results")
    os.makedirs(results_path, exist_ok=True)

    # %% PART B ################################################

    # test learning rates with a 5x5 kernel

    filter_size = 5
    num_filters = 16
    batch_size = 64

    epochs = 30

    for i in range(1, 5):

        lr = 10**-i
        name = "partb_lr_1e-" + str(i)

        krmodel = create_the_model(x_train, y_train, filter_size, num_filters)

        krmodel.train(x_train, y_train, x_valid, y_valid, lr, epochs,
                      batch_size)

        learning_curve = [1 - a for a in krmodel.valid_accs]

        results_dump(results_path, name, lr, filter_size, num_filters,
                     batch_size, learning_curve)

        print("Model " + name + "saved!")

    # %% PART C ################################################

    # kernel sizes

    lr = 0.01  # worked best before

    for filter_size in (1, 3, 5, 7):

        name = "partc_fs-" + str(filter_size)

        krmodel = create_the_model(x_train, y_train, filter_size, num_filters)

        krmodel.train(x_train, y_train, x_valid, y_valid, lr, epochs,
                      batch_size)

        learning_curve = [1 - a for a in krmodel.valid_accs]

        results_dump(results_path, name, lr, filter_size, num_filters,
                     batch_size, learning_curve)

        print("Model " + name + "saved!")