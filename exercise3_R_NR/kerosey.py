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

# tensor flow likes: [batch, height, width, channels]


def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def make_batches(x, y, batch_size):
    assert x.shape[0] == y.shape[0]
    for i in range(ceil(x.shape[0] / batch_size)):
        yield x[batch_size * i:batch_size *
                (i + 1), ...], y[batch_size * i:batch_size * (i + 1), ...]


class Kerosey():
    """
    my ghetto implementation of a usable API for tensorflow. enjoy.

    main limitations: 
        - uses the default tf Graph, so you can't usefully create more than
        one model at a time.

        - only the layer types needed for this exercise have been defined, but you can
        play with them / rearrange them as you like.
    """

    class Layer():
        def __init__(self):
            self.shape = None
            self.func = None

    class InputLayer(Layer):
        def __init__(self, x_shape):
            assert len(
                x_shape
            ) == 4, "Input should be 4D: [batch, height, width, channels]"
            self.shape = x_shape
            self.func = lambda x: x

    class MultiplierLayer(Layer):
        def __init__(self, model, const):
            self.shape = model.layers[-1].shape
            self.func = lambda x: const * x

    class Conv2D(Layer):
        def __init__(self, model, filter_size, num_filters, stride, padding='SAME'):
            last_layer = model.layers[-1]
            # depth of input gets replaced with num_filters
            assert len(
                last_layer.shape
            ) == 4, "Input should be 4D: [batch, height, width, channels]"
            
            if padding == 'SAME':
                self.shape = last_layer.shape[:-1] + (num_filters,)  # replace c with d
            elif padding == 'VALID':
                self.shape = (last_layer.shape[0], last_layer.shape[1] - filter_size + 1, last_layer.shape[2] - filter_size + 1, num_filters)

            # stride not working, obviously.

            # for weights, tf likes [height, width, in, out]
            # Q: how will we force weight sharing between input layers?
            self.weights = tf.get_variable(
                'w' + str(len(model.layers)),
                shape=(filter_size, filter_size, last_layer.shape[-1],
                       num_filters),
                initializer=tf.contrib.layers.xavier_initializer())

            self.bias = tf.get_variable(
                'b' + str(len(model.layers)),
                shape=(num_filters, ),
                initializer=tf.contrib.layers.xavier_initializer())

            self.func = lambda x: tf.nn.bias_add(
                tf.nn.conv2d(
                    x, self.weights, strides=[
                        1, stride, stride, 1], padding=padding), self.bias
            )

    class Conv3D(Layer):
        """we don't take num_filters here: the #outputs will be 
        identical to the so-called input depth"""

        def __init__(self, model, filter_size, num_filters, stride):
            last_layer = model.layers[-1]
            assert len(
                last_layer.shape
            ) == 4, "Input should be 4D: [batch, height, width, channels]"
            self.shape = last_layer.shape[:-1] + (
                num_filters * last_layer.shape[-1], )

            # Weights are now [Dep x Wid x Hgt x in x out]
            # 'in' will always be 1 here: we convert in_channels to depth
            # likewise, the dimension in depth will be 1, because we want shared weights across
            # the depth of the history
            # this comment will not make sense tomorrow
            self.weights = tf.get_variable(
                'w' + str(len(model.layers)),
                shape=(1, filter_size, filter_size, 1, num_filters),
                initializer=tf.contrib.layers.xavier_initializer())

            # self.bias = tf.get_variable(
            # 'b' + str(len(model.layers)),
            # shape=(1, ),
            # initializer=tf.contrib.layers.xavier_initializer())

            # reshape x from [batch, wid, hgt, in]
            # to [batch, dep=in, wid, hgt, 1]

            def conv3d(x):
                # move input channel to depth position; create new dummy input channel to replace it
                # we now have a 5D input [batch, dep, wid, hgt, c]
                x = tf.expand_dims(tf.transpose(x, perm=[0, 3, 1, 2]), -1)

                # convolve it
                # note: first and last stride (batch/channel) are "always 1" (thanks tensorflow)
                # second 1 is image depth
                x = tf.nn.conv3d(
                    x,
                    self.weights,
                    strides=[1, 1, stride, stride, 1],
                    padding='SAME')

                # so now we have [batch, dep, wid, hgt, c']
                # reshape to [batch, wid, hgt, c' * depth]
                # so that we can use 2d convolutions from now on
                x = tf.transpose(x, perm=[0, 2, 3, 1, 4])
                x_shape = tf.shape(x)
                x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[2], -1])
                return x

            self.func = conv3d

    class MaxPool(Layer):
        def __init__(self, model, pool_size):
            last_layer = model.layers[-1]
            assert len(
                last_layer.shape
            ) == 4, "Input should be 4D: [batch, height, width, channels]"
            self.shape = (last_layer.shape[0],
                          ceil(last_layer.shape[1] / pool_size),
                          ceil(last_layer.shape[2] / pool_size),
                          last_layer.shape[3])
            pool_shape = (1, pool_size, pool_size, 1)

            self.func = lambda x: tf.nn.max_pool(
                x, ksize=pool_shape,
                strides=pool_shape, padding='SAME')

    class Relu(Layer):
        def __init__(self, model):
            last_layer = model.layers[-1]
            self.shape = last_layer.shape
            self.func = lambda x: tf.nn.elu(x)

    class Dropout(Layer):
        def __init__(self, model, drop_prob):
            last_layer = model.layers[-1]
            self.shape = last_layer.shape
            self.func = lambda x: tf.nn.dropout(x, 1 - drop_prob)  # tf.nn.drop takes keep_prob as its argument

    class Flatten(Layer):
        def __init__(self, model):
            last_layer = model.layers[-1]

            # nultiply out dimensions except the first (batch) one
            self.shape = (last_layer.shape[0],
                          reduce(lambda x, y: x * y, last_layer.shape[1:], 1))
            self.func = lambda x: tf.reshape(x, (-1, ) + self.shape[1:])

    class Dense(Layer):
        def __init__(self, model, num_units):
            last_layer = model.layers[-1]
            # nultiply out dimensions except the first (batch) one
            self.shape = (last_layer.shape[0], num_units)
            self.weights = tf.get_variable(
                'w' + str(len(model.layers)),
                shape=(last_layer.shape[-1], num_units),
                initializer=tf.contrib.layers.xavier_initializer())
            self.bias = tf.get_variable(
                'b' + str(len(model.layers)),
                shape=(num_units, ),
                initializer=tf.contrib.layers.xavier_initializer())

            self.func = lambda x: tf.add(tf.matmul(x, self.weights), self.bias)

    class LayeredModel(object):
        def __init__(self):
            self.layers = []
            tf.reset_default_graph()  #seems like a good time to do this
            self.tfp_x = None
            self.tfp_y = None
            self.loss_fn = None
            self.optimiser = None
            self.session = None
            self.tfp_drop_prob = tf.placeholder_with_default(0.0, shape=())

        def setup_input(self, x_shape, y_shape):

            x_shape = (None, ) + x_shape[1:]
            y_shape = (None, ) + y_shape[1:]

            self.layers = [Kerosey.InputLayer(x_shape)]

            self.tfp_x = tf.placeholder("float", x_shape)
            self.tfp_y = tf.placeholder("float", y_shape)

        def add_layer(self, layer_type, **kwargs):
            self.layers.append(layer_type(model=self, **kwargs))

        def compile(self, loss, optimiser, learning_rate):
            self.predictor = reduce(lambda f, g: lambda x: g(f(x)),
                                    [l.func for l in self.layers], lambda x: x)

            if loss == 'crossentropy':
                self.loss_fn = tf.losses.softmax_cross_entropy
            elif loss == 'mse':
                self.loss_fn = tf.losses.mean_squared_error
            else:
                raise ("unknown loss")

            if optimiser == 'sgd':
                self.optimiser = tf.train.GradientDescentOptimizer(
                    learning_rate=learning_rate)
            elif optimiser == 'rmsprop':
                self.optimiser = tf.train.RMSPropOptimizer(
                    learning_rate=learning_rate)
            elif optimiser == 'adam':
                self.optimiser = tf.train.AdamOptimizer(
                    learning_rate=learning_rate)
            else:
                raise ("unknown optimiser")

            _ = self.optimiser.minimize(self.loss())

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.local_variables_initializer())
            # self.session = tf.Session()
            # self.session.run(tf.global_variables_initializer()) #why both? who knows

            self.train_losses = []
            self.train_accs = []
            self.valid_losses = []
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

        def run(self, v, x, y=None):
            feed_dict = {self.tfp_x: x}

            if y is not None:
                feed_dict[self.tfp_y] = y

            return self.session.run(v, feed_dict=feed_dict)

        def crunch(self, x, y, batch_size, results, collect_stats=False):
            rr = None  #this will be a list

            for x_batch, y_batch in make_batches(x, y, batch_size):

                r = self.run(results, x_batch, y_batch)

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

        def train(self,
                  x_train,
                  y_train,
                  x_valid,
                  y_valid,
                  epochs,
                  batch_size,
                  tensorboard_dir=None):

            if tensorboard_dir is not None:
                tf_summ_train_loss = tf.summary.scalar('TrainingLoss',
                                                       self.loss())
                tf_summ_train_acc = tf.summary.scalar('TrainingAccuracy',
                                                      self.accuracy())
                tf_summ_valid_loss = tf.summary.scalar('ValidationLoss',
                                                       self.loss())
                tf_summ_valid_acc = tf.summary.scalar('ValidationAccuracy',
                                                      self.accuracy())
                tf_summ_writer = tf.summary.FileWriter(
                    os.path.join('.', tensorboard_dir), self.session.graph)

            train_loss = self.crunch(x_train, y_train, batch_size,
                                     [self.loss()], True)

            print("Initial loss:", train_loss)

            for i in range(epochs):

                self.crunch(x_train, y_train, batch_size,
                            self.optimiser.minimize(self.loss()))

                train_loss, train_acc = self.crunch(
                    x_train, y_train, batch_size,
                    [self.loss(), self.accuracy()], True)

                valid_loss, valid_acc = self.crunch(
                    x_valid, y_valid, batch_size,
                    [self.loss(), self.accuracy()], True)

                if tensorboard_dir is not None:
                    tf_summ_train_loss_c, tf_summ_train_acc_c = self.run(
                        [tf_summ_train_loss, tf_summ_train_acc], x_train,
                        y_train)
                    tf_summ_valid_loss_c, tf_summ_valid_acc_c = self.run(
                        [tf_summ_valid_loss, tf_summ_valid_acc], x_valid,
                        y_valid)
                    tf_summ_writer.add_summary(tf_summ_train_loss_c, i)
                    tf_summ_writer.add_summary(tf_summ_train_acc_c, i)
                    tf_summ_writer.add_summary(tf_summ_valid_loss_c, i)
                    tf_summ_writer.add_summary(tf_summ_valid_acc_c, i)

                print("Iter " + str(i) + ", Loss= " +
                      "{:.4f}".format(train_loss) + ", Train. Acc= " +
                      "{:.4f}".format(train_acc) +
                      ", Val. Acc={:.4f}".format(valid_acc))

                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                self.valid_losses.append(valid_loss)
                self.valid_accs.append(valid_acc)

            print("Training finished")

        def test(self, x_test, y_test, batch_size=1024):
            self.test_acc = self.crunch(x_test, y_test, batch_size,
                                        [self.accuracy()], True)
            print("Test Accuracy {:4f}".format(self.test_acc))
            pass

        def save(self, directory):

            dirpath = os.path.join('.', directory)

            # .create result and model folders
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)

            tf.train.Saver().save(
                self.session, os.path.join(dirpath, 'model.ckpt'))
            print("Model saved in path: %s" % directory)
            pass

        def restore(self, directory):
            tf.train.Saver().restore(
                self.session, os.path.join('.', directory, 'model.ckpt'))
            print("Model restored from path: %s" % directory)
            pass
