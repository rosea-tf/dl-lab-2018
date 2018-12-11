# %% Definitions

import numpy as np
import gym
import itertools as it
from dqn.dqn_agent import DQNAgent
from tensorboard_evaluation import Evaluation
from dqn.networks import NeuralNetwork, TargetNetwork
from utils import EpisodeStats
import os
import json

#%%

MODEL_TEST_INTERVAL = 10 # after this number of episodes, test agent with deterministic actions
MODEL_SAVE_INTERVAL = 100 # yep


# %%


def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()
    
    step = 0
    while True:

        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, model_dir):

    ckpt_dir = os.path.join(model_dir, "ckpt")
    tensorboard_dir = os.path.join(model_dir, "tb")
    
    for d in [model_dir, ckpt_dir, tensorboard_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

    print("... train agent")

    # this might be a dumb idea -- doesn't check that models are the same
    # if os.path.exists(os.path.join(ckpt_dir, 'checkpoint')):
    #     print ("...existing model found! loading")
    #     agent.load(os.path.join(ckpt_dir, 'dqn_agent.ckpt'))
            
    # TODO: make this better
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), [
                             "episode_reward", "left", "right"])
    tensorboard_test = Evaluation(os.path.join(tensorboard_dir, "test"), [
                                  "episode_reward", "left", "right"])

    # training
    for i in range(num_episodes):
        print("episode: ", i)
        stats = run_episode(env, agent, deterministic=False, do_training=True)
        tensorboard.write_episode_data(i, eval_dict={"episode_reward": stats.episode_reward,
                                                     "left": stats.get_action_usage(0),
                                                     "right": stats.get_action_usage(1)})


        if i % MODEL_TEST_INTERVAL == 0 or i >= (num_episodes - 1):
            # evaluate your agent once in a while for some episodes using run_episode(env, agent, deterministic=True, do_training=False) to
            # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.

            stats_test = run_episode(
                env, agent, deterministic=True, do_training=False)

            tensorboard_test.write_episode_data(i, eval_dict={"episode_reward": stats_test.episode_reward,
                                                              "left": stats_test.get_action_usage(0),
                                                              "right": stats_test.get_action_usage(1)})

        # store model every 100 episodes and in the end.
        if i % MODEL_SAVE_INTERVAL == 0 or i >= (num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(
                ckpt_dir, "dqn_agent.ckpt"))


    tensorboard.close_session()
    tensorboard_test.close_session()
#%%
def make_cartpole_agent(name, hidden=20, lr=1e-4, discount_factor=0.99, batch_size=64, 
                        epsilon=0.1, tau=0.01, double_q=False, eps_method=None, save_hypers=True):

    # 1. init Q network and target network (see dqn/networks.py)
    state_dim = 4  # set by cartpole
    num_actions = 2 #set by cartpole

    hypers = locals()

    # prepare a model folder
    base_path = os.path.join('.', 'cartpole')
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    model_path = os.path.join(base_path, name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # save hypers into folder -- used for reconstructing model at test time
    if save_hypers:
        with open(os.path.join(model_path, "hypers.json"), "w") as fh:
            json.dump(hypers, fh)
            
    Q_current = NeuralNetwork(state_dim, num_actions, hidden, lr)
    
    Q_target = TargetNetwork(state_dim, num_actions, hidden, lr, tau)

    # 2. init DQNAgent (see dqn/dqn_agent.py)
    agent = DQNAgent(name, Q_current, Q_target, num_actions,
                     discount_factor, batch_size, epsilon, double_q, eps_method)
    
    return agent, model_path


if __name__ == "__main__":

    # You find information about cartpole in
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("CartPole-v0").unwrapped
    
    agent, model_path = make_cartpole_agent('basic')

    # 3. train DQN agent with train_online(...)
    train_online(env, agent, num_episodes=20, model_dir=model_path)
