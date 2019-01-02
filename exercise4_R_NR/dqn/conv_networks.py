import tensorflow as tf
import numpy as np


class CNN():
    """
    Neural Network class based on TensorFlow.
    """

    def __init__(self,
                 num_actions,
                 lr=1e-4,
                 history_length=0,
                 diff_history=False,
                 big=False):
        
        self.big = big  # implements largeness

        self._build_model(num_actions, lr, history_length, diff_history)
        
    def _build_model(self, num_actions, lr, history_length, diff_history):
        """
        This method creates a neural network with two hidden fully connected layers and 20 neurons each. The output layer
        has #a neurons, where #a is the number of actions and has linear activation.
        Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with a learning rate of 1e-4).
        """

        self.diff_history = diff_history

        if not diff_history:
            self.states_ = tf.placeholder(
                tf.float32, shape=[None, 96, 96, history_length + 1])
        else:
            self.states_ = tf.placeholder(tf.float32, shape=[None, 96, 96, 1])
            self.hstates_ = tf.placeholder(
                tf.float32, shape=[None, 96, 96, history_length])

        self.actions_ = tf.placeholder(
            tf.int32, shape=[None])  # Integer id of which action was selected
        self.targets_ = tf.placeholder(
            tf.float32, shape=[None])  # The TD target value

        fc1_size = 512 if not self.big else 768
        fc2_size = 128 if not self.big else 192

        mu = 0
        sigma = 0.1

        # filter sizes
        fs1, fs2, fs3 = (7, 5, 3)

        #filter counts
        nf1, nf2, nf3 = (16, 32, 48) if not self.big else (32, 48, 64)
        
        #expand network a bit, if we're using history
        if history_length > 0:
            if not diff_history:
                #images are processed in a single convnet, so...
                # 50% more layer 1 filters for every additional frame processed
                nf1 = int(nf1 * (1 + (history_length * 0.5)))
                # 25% more layer 2 filters for every additional frame processed
                nf2 = int(nf2 * (1 + (history_length * 0.25)))
                
            else:
                # original convnet processes only one frame, so we leave its size unchanged
                
                #filter sizes and counts for subtracted history images
                hfs1, hfs2, hfs3 = (7, 5, 3)
                hnf1, hnf2, hnf3 = (8, 16, 24) if not self.big else (32, 48, 64)
                
                if history_length > 1:
                    # 50% more layer 1 filters for every additional frame processed
                    hnf1 = int(hnf1 * (1 + ((history_length - 1) * 0.5)))
                    # 25% more layer 2 filters for every additional frame processed
                    hnf2 = int(hnf2 * (1 + ((history_length - 1) * 0.25)))

        # first conv layer
        if not diff_history:
            conv1_w = tf.Variable(
                tf.truncated_normal(
                    shape=[fs1, fs1, history_length + 1, nf1],
                    mean=mu,
                    stddev=sigma),
                name="w1")
        else:
            conv1_w = tf.Variable(
                tf.truncated_normal(
                    shape=[fs1, fs1, 1, nf1], mean=mu, stddev=sigma),
                name="w1")

        conv1_b = tf.Variable(tf.zeros(nf1), name="b1")
        conv1 = tf.nn.conv2d(
            self.states_, conv1_w, strides=[1, 1, 1, 1],
            padding='SAME') + conv1_b
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(
            conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # second conv layer
        conv2_w = tf.Variable(
            tf.truncated_normal(
                shape=[fs2, fs2, nf1, nf2], mean=mu, stddev=sigma),
            name="w2")
        conv2_b = tf.Variable(tf.zeros(nf2), name="b2")
        conv2 = tf.nn.conv2d(
            pool1, conv2_w, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(
            conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # third conv layer
        conv3_w = tf.Variable(
            tf.truncated_normal(
                shape=[fs3, fs3, nf2, nf3], mean=mu, stddev=sigma),
            name="w3")
        conv3_b = tf.Variable(tf.zeros(nf3), name="b3")
        conv3 = tf.nn.conv2d(
            pool2, conv3_w, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
        conv3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(
            conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # flatten
        shape = pool3.get_shape().as_list()
        dim = np.prod(shape[1:])
        flat = tf.reshape(pool3, [-1, dim])

        if diff_history:
            # first conv layer
            hconv1_w = tf.Variable(
                tf.truncated_normal(
                    shape=[hfs1, hfs1, history_length, hnf1],
                    mean=mu,
                    stddev=sigma),
                name="hw1")
            hconv1_b = tf.Variable(tf.zeros(hnf1), name="hb1")
            hconv1 = tf.nn.conv2d(
                self.hstates_, hconv1_w, strides=[1, 1, 1, 1],
                padding='SAME') + hconv1_b
            hconv1 = tf.nn.relu(hconv1)
            hpool1 = tf.nn.max_pool(
                hconv1,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME')

            # second conv layer
            hconv2_w = tf.Variable(
                tf.truncated_normal(
                    shape=[hfs2, hfs2, hnf1, hnf2], mean=mu, stddev=sigma),
                name="hw2")
            hconv2_b = tf.Variable(tf.zeros(hnf2), name="hb2")
            hconv2 = tf.nn.conv2d(
                hpool1, hconv2_w, strides=[1, 1, 1, 1],
                padding='SAME') + hconv2_b
            hconv2 = tf.nn.relu(hconv2)
            hpool2 = tf.nn.max_pool(
                hconv2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME')

            # third conv layer
            hconv3_w = tf.Variable(
                tf.truncated_normal(
                    shape=[hfs3, hfs3, hnf2, hnf3], mean=mu, stddev=sigma),
                name="hw3")
            hconv3_b = tf.Variable(tf.zeros(hnf3), name="hb3")
            hconv3 = tf.nn.conv2d(
                hpool2, hconv3_w, strides=[1, 1, 1, 1],
                padding='SAME') + hconv3_b
            hconv3 = tf.nn.relu(hconv3)
            hpool3 = tf.nn.max_pool(
                hconv3,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME')

            # flatten
            hshape = hpool3.get_shape().as_list()
            hdim = np.prod(hshape[1:])
            hflat = tf.reshape(hpool3, [-1, hdim])

            # cat it onto the flattened main output
            flat = tf.concat([flat, hflat], axis=1)

        # network
        fc1 = tf.layers.dense(flat, fc1_size, tf.nn.relu)
        fc2 = tf.layers.dense(fc1, fc2_size, tf.nn.relu)
        self.predictions = tf.layers.dense(fc2, num_actions)

        # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.states_)[0]
        gather_indices = tf.range(batch_size) * tf.shape(
            self.predictions)[1] + self.actions_
        self.action_predictions = tf.gather(
            tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.targets_,
                                            self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """
        if not self.diff_history:
            prediction = sess.run(self.predictions, {self.states_: states})
        else:
            prediction = sess.run(self.predictions, {
                self.states_: states[..., 0:1],
                self.hstates_: states[..., 1:]
            })
        return prediction

    def update(self, sess, states, actions, targets):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """
        if not self.diff_history:
            feed_dict = {
                self.states_: states,
                self.targets_: targets,
                self.actions_: actions
            }
        else:
            feed_dict = {
                self.states_: states[..., 0:1],
                self.hstates_: states[..., 1:],
                self.targets_: targets,
                self.actions_: actions
            }

        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        return loss


class CNNTargetNetwork(CNN):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self,
                 num_actions,
                 lr=1e-4,
                 tau=0.01,
                 history_length=0,
                 diff_history=False,
                 big=False):

        # check that the _register_associates method won't fuck us up
        num_vars_before = len(tf.trainable_variables())

        CNN.__init__(self, num_actions, lr, history_length,
                     diff_history, big)

        num_vars_after = len(tf.trainable_variables())

        assert num_vars_after == 2 * num_vars_before

        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx, var in enumerate(tf_vars[0:total_vars // 2]):
            op_holder.append(tf_vars[idx + total_vars // 2].assign(
                (var.value() * self.tau) + (
                    (1 - self.tau) * tf_vars[idx + total_vars // 2].value())))
        return op_holder

    def update(self, sess):
        for op in self._associate:
            sess.run(op)
