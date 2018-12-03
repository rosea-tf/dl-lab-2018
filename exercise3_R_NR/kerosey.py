from __future__ import print_function

import os

import numpy as np

import tensorflow as tf

from math import ceil

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

    class Sequentialise(Layer):
        """
        this takes a [batch, width, height, channels] tensor
        and converts it [timeseq(=channel), batch, width, height, 1] tensor
        so that historical input images are kept separate in subsequent convolutions
        """

        def __init__(self, model):
            last_layer = model.layers[-1]
            assert len(
                last_layer.shape
            ) == 4, "Input should be 4D: [batch, height, width, channels]"
            self.shape = (last_layer.shape[3], *last_layer.shape[0:3], 1)
            self.func = lambda x: tf.expand_dims(tf.transpose(x, perm=[3, 0, 1, 2]), -1)

    class Conv2D(Layer):
        def __init__(self,
                     model,
                     filter_size,
                     num_filters,
                     stride,
                     padding='SAME'):
            last_layer = model.layers[-1]
            # depth of input gets replaced with num_filters
            assert len(
                last_layer.shape
            ) == 4, "Input should be 4D: [batch, height, width, channels]"

            if padding == 'SAME':
                self.shape = last_layer.shape[:-1] + (num_filters,
                                                      )  # replace c with d
            elif padding == 'VALID':
                self.shape = (last_layer.shape[0],
                              last_layer.shape[1] - filter_size + 1,
                              last_layer.shape[2] - filter_size + 1,
                              num_filters)

            # stride not working, obviously.

            # for weights, tf likes [height, width, in, out]
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

    class ConvSeq2D(Layer):
        def __init__(self,
                     model,
                     filter_size,
                     num_filters,
                     stride,
                     padding='SAME'):
            last_layer = model.layers[-1]
            assert len(
                last_layer.shape
            ) == 5, "Input should be 5D: [timeseq, batch, height, width, channels]"

            if padding == 'SAME':
                self.shape = last_layer.shape[:-1] + (num_filters,
                                                      )  # replace c with d
            elif padding == 'VALID':
                self.shape = (last_layer.shape[0], last_layer.shape[1],
                              last_layer.shape[2] - filter_size + 1,
                              last_layer.shape[3] - filter_size + 1,
                              num_filters)

            # stride not working, obviously.

            # weights -- same as basic 2d case
            self.weights = tf.get_variable(
                'w' + str(len(model.layers)),
                shape=(filter_size, filter_size, last_layer.shape[-1],
                       num_filters),
                initializer=tf.contrib.layers.xavier_initializer())

            self.bias = tf.get_variable(
                'b' + str(len(model.layers)),
                shape=(num_filters, ),
                initializer=tf.contrib.layers.xavier_initializer())

            def convseq2d(x):
                # iterate over timeseq
                result_stack = []

                for t in range(self.shape[0]):
                    result = tf.nn.bias_add(
                        tf.nn.conv2d(
                            x[t],
                            self.weights,
                            strides=[1, stride, stride, 1],
                            padding=padding), self.bias)
                    result_stack.append(result)

                return tf.stack(result_stack, axis=0)

            self.func = convseq2d

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

    class MaxPoolSeq(Layer):
        """tf demands 4d for maxpool, so it seems we have to wrap the 5d case..."""

        def __init__(self, model, pool_size):
            last_layer = model.layers[-1]
            assert len(last_layer.shape) == 5, "Input should be 5D"

            self.shape = (last_layer.shape[0], last_layer.shape[1],
                          ceil(last_layer.shape[2] / pool_size),
                          ceil(last_layer.shape[3] / pool_size),
                          last_layer.shape[4])

            pool_shape = (1, pool_size, pool_size, 1)

            def maxpoolseq(x):
                # iterate over timeseq
                result_stack = []

                for t in range(self.shape[0]):
                    result = tf.nn.max_pool(
                        x[t],
                        ksize=pool_shape,
                        strides=pool_shape,
                        padding='SAME')
                    result_stack.append(result)

                return tf.stack(result_stack, axis=0)

            self.func = maxpoolseq

    class Relu(Layer):
        def __init__(self, model):
            last_layer = model.layers[-1]
            self.shape = last_layer.shape
            self.func = lambda x: tf.nn.elu(x)

    class Dropout(Layer):
        def __init__(self, model, drop_prob):
            last_layer = model.layers[-1]
            self.shape = last_layer.shape
            model.val_drop_prob = drop_prob  # this is ugly but it will do.
            # means that layers will all share the same drop prob.

            # tf.nn.drop takes keep_prob as its argument, so we do (1 - p)
            self.func = lambda x: tf.nn.dropout(x, 1 - model.tfp_drop_prob)

    class Flatten(Layer):
        def __init__(self, model):
            last_layer = model.layers[-1]

            # nultiply out dimensions except the first (batch) one
            self.shape = (last_layer.shape[0],
                          reduce(lambda x, y: x * y, last_layer.shape[1:], 1))
            self.func = lambda x: tf.reshape(x, (-1, ) + self.shape[1:])

    class FlattenSeq(Layer):
        """
        this takes a [timeseq, batch, width, height, channels] tensor
        and converts it [batch, *]
        for use when we're ready for dense layers, and not using an LSTM
        """

        def __init__(self, model, keep_seq=False):
            last_layer = model.layers[-1]
            assert len(last_layer.shape) == 5, "Input should be 5D"
            # nultiply out dimensions except the first (batch) one

            if keep_seq:
                self.shape = (last_layer.shape[0], last_layer.shape[1],
                              reduce(lambda x, y: x * y, last_layer.shape[2:],
                                     1))
                # keep first two dimensions intact
                self.func = lambda x: tf.reshape(x, (self.shape[0], -1) + self.shape[2:])
                # this format is inconsistent with the format below, but it will do
            else:
                self.shape = (last_layer.shape[1],
                              last_layer.shape[0] * reduce(
                                  lambda x, y: x * y, last_layer.shape[2:], 1))
                self.func = lambda x: tf.reshape(tf.transpose(x, perm=[1, 0, 2, 3, 4]), (-1, ) + self.shape[1:])

    class Dense(Layer):
        def __init__(self, model, num_units):
            last_layer = model.layers[-1]
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

    class LSTM(Layer):
        """The LSTM layer. A sequence -> single value setup"""

        # should just wrap it all up into one class, I think.

        def __init__(self, model,
                     num_units):  # input_size: int, hidden_size: int):

            # C and H will have the same size
            last_layer = model.layers[-1]

            assert len(
                last_layer.shape
            ) == 3, "last layer should have shape: [timeseq, batch, feature]"

            last_layer = model.layers[-1]
            self.shape = (last_layer.shape[1], num_units)
            self.weight_shape = (last_layer.shape[2] + num_units,
                                 num_units)  #in, out

            # note: c and h have the same size
            def gate_wt_gen(code):
                """generates weight and bias variables for use in each of the four gates"""

                w = tf.get_variable(
                    'w' + str(len(model.layers)) + code,
                    shape=(last_layer.shape[2] + num_units, num_units),
                    initializer=tf.contrib.layers.xavier_initializer())

                b = tf.get_variable(
                    'b' + str(len(model.layers)) + code,
                    shape=(num_units, ),
                    initializer=tf.constant_initializer(
                        0.8))  # for gates - initialise as (mostly) open

                return w, b

            self.forget_gate_w, self.forget_gate_b = gate_wt_gen('f')
            self.input_gate_w, self.input_gate_b = gate_wt_gen('i')
            self.candidate_w, self.candidate_b = gate_wt_gen('c')
            self.output_gate_w, self.output_gate_b = gate_wt_gen('o')

            def lstm(x):

                # initialise C and H cell states (for use in first timestep)
                h = tf.einsum('i,j->ij', tf.zeros_like(x[0, :, 0]),
                              tf.zeros((self.shape[-1])))
                c = tf.einsum('i,j->ij', tf.zeros_like(x[0, :, 0]),
                              tf.zeros((self.shape[-1])))
                # is there a better way of doing this, since tf doesn't know x's batch dimension?

                seq_len = x.shape[0]

                for t in range(seq_len):
                    # x[t] will be [batch, in_features]; h is [batch, out_features(=num_units)]
                    xh = tf.concat([x[t], h], axis=1)

                    # calculate and apply forget gate
                    # it should really be called a 'remember gate'.
                    forget_gate = tf.sigmoid(
                        tf.add(
                            tf.matmul(xh, self.forget_gate_w),
                            self.forget_gate_b))
                    c = tf.multiply(forget_gate, c)

                    # calculate and apply input gate, candidate input
                    input_gate = tf.sigmoid(
                        tf.add(
                            tf.matmul(xh, self.input_gate_w),
                            self.input_gate_b))
                    candidate = tf.tanh(
                        tf.add(
                            tf.matmul(xh, self.candidate_w), self.candidate_b))
                    c = tf.add(c, tf.multiply(candidate, input_gate))

                    # decide what to output
                    output_gate = tf.sigmoid(
                        tf.add(
                            tf.matmul(xh, self.output_gate_w),
                            self.output_gate_b))
                    h = tf.multiply(tf.tanh(c), output_gate)

                return h

            self.func = lstm

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
            self.val_drop_prob = 0.0

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

        def run(self, v, x, y=None, train_mode=False):
            feed_dict = {self.tfp_x: x}

            if y is not None:
                feed_dict[self.tfp_y] = y

            if train_mode and self.val_drop_prob > 0:
                feed_dict[self.tfp_drop_prob] = self.val_drop_prob
            else:
                feed_dict[self.tfp_drop_prob] = 0.0

            return self.session.run(v, feed_dict=feed_dict)

        def crunch(self,
                   x,
                   y,
                   batch_size,
                   results,
                   train_mode=False,
                   collect_stats=False):
            rr = None  #this will be a list

            for x_batch, y_batch in make_batches(x, y, batch_size):

                r = self.run(results, x_batch, y_batch, train_mode)

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

            train_loss = self.crunch(
                x_train,
                y_train,
                batch_size, [self.loss()],
                train_mode=True,
                collect_stats=True)

            print("Initial loss:", train_loss)

            for i in range(epochs):

                self.crunch(
                    x_train,
                    y_train,
                    batch_size,
                    self.optimiser.minimize(self.loss()),
                    collect_stats=False)

                # dropout (if any) activated (via train_mode)
                train_loss, train_acc = self.crunch(
                    x_train,
                    y_train,
                    batch_size, [self.loss(), self.accuracy()],
                    train_mode=True,
                    collect_stats=True)

                # dropout deactivated
                valid_loss, valid_acc = self.crunch(
                    x_valid,
                    y_valid,
                    batch_size, [self.loss(), self.accuracy()],
                    train_mode=False,
                    collect_stats=True)

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
            self.test_acc = self.crunch(
                x_test,
                y_test,
                batch_size, [self.accuracy()],
                train_mode=False,
                collect_stats=True)
            print("Test Accuracy {:4f}".format(self.test_acc))


        def save(self, directory):

            dirpath = os.path.join('.', directory)

            # .create result and model folders
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)

            tf.train.Saver().save(self.session,
                                  os.path.join(dirpath, 'model.ckpt'))
            print("Model saved in path: %s" % directory)

        def restore(self, directory):
            tf.train.Saver().restore(
                self.session, os.path.join('.', directory, 'model.ckpt'))
            print("Model restored from path: %s" % directory)
