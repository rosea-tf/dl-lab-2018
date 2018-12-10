import tensorflow as tf
import numpy as np
from dqn.replay_buffer import ReplayBuffer

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, discount_factor=0.99, batch_size=64, epsilon=0.05, double_q=False):
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
        self.Q = Q      
        self.Q_target = Q_target
        
        self.epsilon = epsilon

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.discount_factor = discount_factor

        self.double_q = double_q

        # define replay buffer
        self.replay_buffer = ReplayBuffer()

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
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        

        # 2. sample next batch
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(self.batch_size)
        
        # find optimal actions for the sampled s' states
        if self.double_q:
            # double Q learning (select actions using current network, rather than target network)
            # ...in order to decorrelate noise between selection and evaluation
            # (Q(state,action) is still evaluated using target network in any case)
            action_selector = self.Q
        else:
            action_selector = self.Q_target
            

        # as usual, the Q network returns a vector of... predicted values for every possible action
        a_prime = np.argmax(action_selector.predict(self.sess, batch_next_states), axis=1)
        
        # pick a''th value from each column of the Q prediction
        # note, this will include action predictions for "done" state, but we'll kill them later
        q_values_next = self.Q.predict(self.sess, batch_next_states)[np.arange(self.batch_size), a_prime]
        
        # 2.1 compute td targets: 
        # if done, there will be no next state
        td_targets = batch_rewards + np.where(batch_dones, 0, discount * q_values_next)
        
        
        # 2.2 update the Q (current) network
        self.Q.update(self, self.sess, batch_states, batch_actions, td_targets)
        
        
        # 2.3 call soft update for target network
            # this is done by the dodgy associate_method therein
        self.Q_target.update(self, self.sess)
   

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
        action_probs = action_selector.predict(self.sess, batch_next_states)
        
        
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # take greedy action
            action_id = np.argmax(action_probs, axis=1)
        else:

            # sample random action

            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.

            action_id = np.random.choice(np.arange(len(action_probs)), p=action_probs)
          
        return action_id


    def load(self, file_name):
        self.saver.restore(self.sess, file_name)
