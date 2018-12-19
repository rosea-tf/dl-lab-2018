import tensorflow as tf
import numpy as np
import os
from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def define_model(x,y):
        pass

    def __init__(self, history_len=1, lr=0.0001, tensorboard_dir="./tensorboard", name="unnamed", dropout=0.75):
        g_1 = tf.Graph()
        with g_1.as_default():
            self.dropout = dropout
            self.history_len = history_len

            # input placeholders
            x = tf.placeholder(tf.float32, shape=[None, 96, 96, history_len])
            y = tf.placeholder(tf.int32, shape=(None,5))

            # call model specific setup
            logits = self.define_model(x,y)

            # cross entropy and loss
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
            loss = tf.reduce_mean(cross_entropy)

            # optimizing / training
            optimizer = tf.train.AdamOptimizer(lr)
            training = optimizer.minimize(loss)

            # softmax for output
            softmax = tf.nn.softmax(logits)

            # measurements
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            # tensorboard output
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('loss', loss)

            # -----------------------------------------
            # keep main components as object attributes
            # -----------------------------------------

            self.merged = tf.summary.merge_all()

            #input nodes
            self.x = x
            self.y = y

            # paths and names
            self.tensorboard_dir = tensorboard_dir
            self.name = name

            # output nodes
            self.loss = loss
            self.softmax = softmax
            self.training = training
            self.accuracy = accuracy

            # Start tensorflow session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            # writers for training and validation
            self.train_writer = tf.summary.FileWriter(os.path.join(tensorboard_dir, name, 'training'))
            self.valid_writer = tf.summary.FileWriter(os.path.join(tensorboard_dir, name, 'validation'))

            # for saving the trained model...
            self.saver = tf.train.Saver()

    def load(self, file_name=None):
        if not file_name:
            file_name = os.path.join('./models', self.name+".ckpt")

        self.saver.restore(self.sess, file_name)

    def save(self, file_name=None):
        if not file_name:
            file_name = os.path.join('./models', self.name+".ckpt")

        self.saver.save(self.sess, file_name)
