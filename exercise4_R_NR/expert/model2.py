import tensorflow as tf
import numpy as np
from model_base import BaseModel

# a simple model with 3 convolutional layers
# scores 800-900 points in test driving using softmax

class Model2(BaseModel):

    def define_model(self, x, y):
        fc1_size = 512
        fc2_size = 128
        fc3_size = 5

        mu = 0
        sigma = 0.1

        # filter sizes
        fs1, fs2, fs3 = (7,5,3)

        #filter counts
        nf1, nf2, nf3 = (16, 32, 48)

        # first conv layer
        conv1_w = tf.Variable(tf.truncated_normal(shape=[fs1, fs1, 1, nf1], mean=mu, stddev=sigma), name="w1")
        conv1_b = tf.Variable(tf.zeros(nf1), name="b1")
        conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # second conv layer
        conv2_w = tf.Variable(tf.truncated_normal(shape=[fs2, fs2, nf1, nf2], mean=mu, stddev=sigma), name="w2")
        conv2_b = tf.Variable(tf.zeros(nf2), name="b2")
        conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # third conv layer
        conv3_w = tf.Variable(tf.truncated_normal(shape=[fs3, fs3, nf2, nf3], mean=mu, stddev=sigma), name="w3")
        conv3_b = tf.Variable(tf.zeros(nf3), name="b3")
        conv3 = tf.nn.conv2d(pool2, conv3_w, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
        conv3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # flatten
        shape = pool3.get_shape().as_list()
        dim = np.prod(shape[1:])
        flat = tf.reshape(pool3, [-1, dim])

        # first linear layer - with dropout and relu
        fc1_w = tf.Variable(tf.truncated_normal(shape=(12*12*48, fc1_size), mean=mu, stddev=sigma), name="fc1_w")
        fc1_b = tf.Variable(tf.zeros(fc1_size), name="fc1_b")
        fc1 = tf.matmul(flat, fc1_w) + fc1_b
        fc1 = tf.nn.dropout(fc1, self.dropout)
        fc1 = tf.nn.relu(fc1)

        # second linear layer - with dropout and relu
        fc2_w = tf.Variable(tf.truncated_normal(shape=(fc1_size, fc2_size), mean=mu, stddev=sigma), name="fc2_w")
        fc2_b = tf.Variable(tf.zeros(fc2_size), name="fc2_b")
        fc2 = tf.matmul(fc1, fc2_w) + fc2_b
        fc2 = tf.nn.dropout(fc2, self.dropout)
        fc2 = tf.nn.relu(fc2)

        # output layer - no dropout or relu here
        fc3_w = tf.Variable(tf.truncated_normal(shape=(fc2_size, fc3_size), mean=mu, stddev=sigma))
        fc3_b = tf.Variable(tf.zeros(fc3_size))
        fc3 = tf.matmul(fc2, fc3_w) + fc3_b

        return fc3



