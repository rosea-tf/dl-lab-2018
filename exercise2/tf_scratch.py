# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:31:12 2018

@author: AlexR
"""


from __future__ import print_function

import gzip
import json
import os
import pickle

import numpy as np
from math import ceil

import tensorflow as tf

from cnn_mnist_solution import mnist
from operator import mul
from functools import reduce


x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(
    os.path.expanduser(os.sep.join(["~", "data"])), recs=1000)

# %%
tf.reset_default_graph()

# To initialize all the variables in TensorFlow, you need to explicitly
# call the global variable intializer global_variables_initializer()

lr = 0.01
num_filters = 5
batch_size = 64
filter_size = 3
epochs = 3  # should be enough

flatlayer_size = 128
pool_size = 2


# placeholders
tf_x = tf.placeholder("float", (None, ) + x_train.shape[1:])
tf_y = tf.placeholder("float", (None, ) + y_train.shape[1:])

# %% Calculate weight sizes

weight_sizes = []
bias_sizes = []
layer_sizes = []

# first weight will be a conv kernel:
# F x F x INPUT_CHANNELS x NUM_FILTERS
# one bias for every filter
# conv='SAME', but input cut down by half because of max pooling operation
weight_sizes.append((filter_size, filter_size, x_train.shape[-1], num_filters))
bias_sizes.append((num_filters,))
layer_sizes.append(
    (-1,
     ceil(
         x_train.shape[1] / pool_size),
        ceil(
            x_train.shape[2] / pool_size),
     num_filters))

# as above, but not we have n_f instead of i_c
weight_sizes.append((filter_size, filter_size, num_filters, num_filters))
bias_sizes.append((num_filters,))
layer_sizes.append((-1,
                    ceil(layer_sizes[-1][1] / pool_size),
                    ceil(layer_sizes[-1][2] / pool_size),
                    num_filters))

# next we have a flattened layer (128), so:
weight_sizes.append(
    (layer_sizes[-1][1] * layer_sizes[-1][2] * num_filters, flatlayer_size))
bias_sizes.append((flatlayer_size,))
layer_sizes.append((-1, flatlayer_size))

# finally, a layer of 10:
weight_sizes.append((flatlayer_size, y_train.shape[1]))
bias_sizes.append((y_train.shape[1],))
layer_sizes.append((-1, y_train.shape[1]))

# set up weights

weights = [
    tf.get_variable(
        'w' + str(i),
        shape=s,
        initializer=tf.contrib.layers.xavier_initializer()) for i,
    s in enumerate(weight_sizes)]

biases = [
    tf.get_variable(
        'b' + str(i),
        shape=s,
        initializer=tf.contrib.layers.xavier_initializer()) for i,
    s in enumerate(bias_sizes)]

# %%


# define a conv2d-relu-maxpool operation
def conv2d(x, w, b, stride=1, pool_size=2):
    # so w is the filter (which will come with a size)
    x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(
        x, ksize=[
            1, pool_size, pool_size, 1], strides=[
            1, pool_size, pool_size, 1], padding='SAME')
    return x


def forward_pass(x, weights, biases):
    conv1 = conv2d(x, weights[0], biases[0])
    conv2 = conv2d(conv1, weights[1], biases[1])

    flat = tf.nn.relu(
        tf.add(
            tf.matmul(
                tf.reshape(
                    conv2,
                    (-1,
                     weight_sizes[2][0])),
                weights[2]),
            biases[2]))

    final = tf.add(tf.matmul(flat, weights[3]), biases[3])

    # guess we do the softmax later
    return final


tf_y_pred = forward_pass(tf_x, weights, biases)

tf_accuracy = tf.reduce_mean(
    tf.cast(
        tf.equal(
            tf.argmax(
                tf_y_pred, 1), tf.argmax(
                    tf_y, 1)), tf.float32))

tf_losses = tf.losses.softmax_cross_entropy(tf_y, tf_y_pred)

tf_loss = tf.reduce_mean(tf_losses)

tf_optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(tf_loss)

# %% do the thing


init = tf.global_variables_initializer()

def make_batches(x, batch_size):
    n_of_data = x.shape[0]
    for i in range(n_of_data//batch_size):
        yield x[batch_size*i:batch_size*(i+1), ...]

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    for i in range(epochs):
        for batch in range(len(x_train) // batch_size):
            batch_x = x_train[batch *
                              batch_size:min((batch + 1) * batch_size, len(x_train))]
            batch_y = y_train[batch
                              * batch_size:min((batch + 1) * batch_size, len(y_train))]
            # Run optimization op (backprop).
            # Calculate batch loss and accuracy
            opt = sess.run(tf_optimizer, feed_dict={tf_x: batch_x,
                                                 tf_y: batch_y})
            loss, acc = sess.run([tf_loss, tf_accuracy], feed_dict={tf_x: batch_x,
                                                              tf_y: batch_y})
        print("Iter " + str(i) + ", Loss= " +
              "{:.6f}".format(loss) + ", Training Accuracy= " +
              "{:.5f}".format(acc))
        
    print("Optimization Finished!")
