import tensorflow as tf
import numpy as np
from dqn.replay_buffer import ReplayBuffer


def softmax(x, temperature=1):  #TODO
    """
    Compute softmax values for each sets of scores in x.
    Used in Boltzmann exploration
    """
    x = np.divide(x, temperature)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class DQNAgent:
    def __init__(self,
                 name,
                 Q_current,
                 Q_target,
                 num_actions,
                 discount_factor,
                 batch_size,
                 epsilon,
                 epsilon_decay,
                 boltzmann,
                 double_q,
                 buffer_capacity,
                 random_probs=None):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            discount_factor: gamma, discount factor of future rewards.
            batch_size: Number of samples per batch.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
        """
        # save hyperparameters in folder

        self.name = name  # probably useless
        self.Q_current = Q_current
        self.Q_target = Q_target

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.boltzmann = boltzmann

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.buffer_capacity = buffer_capacity

        self.double_q = double_q

        self.random_probs = random_probs

        # define replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Start tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        self.replay_buffer.add_transition(state, action, next_state, reward,
                                          terminal)

        # 2. sample next batch
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(
            self.batch_size)

        # find optimal actions for the sampled s' states
        if self.double_q:
            # double Q learning (select actions using current network, rather than target network)
            # ...in order to decorrelate noise between selection and evaluation
            # (Q(state,action) is still evaluated using target network in any case)
            action_selector = self.Q_current
        else:
            action_selector = self.Q_target

        # as usual, the Q network returns a vector of... predicted values for every possible action
        a_prime = np.argmax(
            action_selector.predict(self.sess, batch_next_states), axis=1)

        # pick a''th value from each column of the Q prediction
        # note, this will include action predictions for "done" state, but we'll kill them later
        q_values_next = self.Q_current.predict(
            self.sess, batch_next_states)[np.arange(self.batch_size), a_prime]

        # 2.1 compute td targets:
        # if done, there will be no next state
        td_targets = batch_rewards + np.where(
            batch_dones, 0, self.discount_factor * q_values_next)

        # 2.2 update the Q (current) network
        self.Q_current.update(self.sess, batch_states, batch_actions,
                              td_targets)

        # 2.3 call soft update for target network
        # this is done by the dodgy associate_method therein
        self.Q_target.update(self.sess)

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """

        # get action probabilities from current network
        Q_values = np.squeeze(
            self.Q_current.predict(self.sess, np.expand_dims(state, axis=0)))

        argmax_a = np.argmax(Q_values)

        if deterministic:
            # take greedy action
            return argmax_a

        if self.boltzmann:
            # implementing an interaction here between boltzmann exploration and epsilon:
            # viz. that epsilon controls the temperature of the softmax function
            # so that as before, higher eps -> higher exploration
            action_probs = softmax(
                Q_values, temperature=1 / (1 - self.epsilon)**2)
            action = np.random.choice(
                np.arange(len(action_probs)), p=action_probs)

        else:
            action_probs = np.zeros_like(Q_values)

            if np.random.uniform() > self.epsilon:
                # choose the best action
                action = argmax_a
            else:
                # explore
                if self.random_probs is None:
                    action = np.random.randint(self.num_actions, size=1)[0]

                else:
                    action = np.random.choice(
                        np.arange(self.num_actions), p=self.random_probs)

        # we decay epsilon AFTER we've checked it
        # (nb: if deterministic, epsilon will never decay, but of course this doesn't matter)
        if self.epsilon_decay > 0:
            self.epsilon *= (1 - self.epsilon_decay)

        return action

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)
