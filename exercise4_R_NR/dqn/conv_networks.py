import tensorflow as tf
import numpy as np

# TODO: add your Convolutional Neural Network for the CarRacing environment.

class CNN():
    """
    Neural Network class based on TensorFlow.
    """
    def __init__(self, state_dim, num_actions, hidden=20, lr=1e-4, history_len=1):
        self._build_model(state_dim, num_actions, hidden, lr, history_len)

    def _build_model(self, state_dim, num_actions, hidden, lr, history_len):
        """
        This method creates a neural network with two hidden fully connected layers and 20 neurons each. The output layer
        has #a neurons, where #a is the number of actions and has linear activation.
        Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with a learning rate of 1e-4).
        """

        self.states_ = tf.placeholder(tf.float32, shape=[None, 96, 96, history_len])
        self.actions_ = tf.placeholder(tf.int32, shape=[None])                  # Integer id of which action was selected
        self.targets_ = tf.placeholder(tf.float32,  shape=[None])               # The TD target value

        fc1_size = 512
        fc2_size = 128

        mu = 0
        sigma = 0.1

        # filter sizes
        fs1, fs2, fs3 = (7,5,3)

        #filter counts
        nf1, nf2, nf3 = (16, 32, 48)

        # first conv layer
        conv1_w = tf.Variable(tf.truncated_normal(shape=[fs1, fs1, 1, nf1], mean=mu, stddev=sigma), name="w1")
        conv1_b = tf.Variable(tf.zeros(nf1), name="b1")
        conv1 = tf.nn.conv2d(self.states_, conv1_w, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
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

        # network
        fc1 = tf.layers.dense(flat, fc1_size, tf.nn.relu)
        fc2 = tf.layers.dense(fc1, fc2_size, tf.nn.relu)
        self.predictions = tf.layers.dense(fc2, num_actions)

        # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.states_)[0]
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.targets_, self.action_predictions)
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
        prediction = sess.run(self.predictions, { self.states_: states })
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
        feed_dict = { self.states_: states, self.targets_: targets, self.actions_: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class CNNTargetNetwork(CNN):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """
    def __init__(self, state_dim, num_actions, hidden=20, lr=1e-4, tau=0.01):
        CNN.__init__(self, state_dim, num_actions, hidden, lr)
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx,var in enumerate(tf_vars[0:total_vars//2]):
            op_holder.append(tf_vars[idx+total_vars//2].assign(
              (var.value()*self.tau) + ((1-self.tau)*tf_vars[idx+total_vars//2].value())))
        return op_holder

    def update(self, sess):
        for op in self._associate:
          sess.run(op)
