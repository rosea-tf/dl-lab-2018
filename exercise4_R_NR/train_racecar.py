# %% Definitions

import numpy as np
import gym
import itertools as it
from dqn.dqn_agent import DQNAgent
from tensorboard_evaluation import Evaluation
from dqn.conv_networks import CNN, CNNTargetNetwork
from utils import *
import os
import json

#%%

MODEL_TEST_INTERVAL = 10  # after this number of episodes, test agent with deterministic actions
MODEL_SAVE_INTERVAL = 100  # yep

# %%


def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0


def id_to_action(action_id):
    # determine final action
    a = [0, 0, 0]

    if action_id == 1:
        a[0] = -1
    elif action_id == 2:
        a[0] = 1
    elif action_id == 3:
        a[1] = 1
    elif action_id == 4:
        a[2] = 0.2

    return a


def run_episode(env,
                agent,
                deterministic,
                skip_frames=0,
                do_training=True,
                rendering=False,
                max_timesteps=1000,
                history_length=0):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)

    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)

        action_id = agent.act(state=state, deterministic=deterministic)
        action = id_to_action(action_id)
        next_state, reward, terminal, info = env.step(action)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal:
                break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, model_dir, history_length=0):

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
    tensorboard = Evaluation(
        os.path.join(tensorboard_dir, "train"),
        ["episode_reward", "straight", "left", "right", "accel", "brake"])
    tensorboard_test = Evaluation(
        os.path.join(tensorboard_dir, "test"),
        ["episode_reward", "straight", "left", "right", "accel", "brake"])

    # training
    for i in range(num_episodes):
        print("episode: ", i)

        max_timesteps = min(
            1000, 4 * i + 100)  #adr - sped up compared to ingmar's version

        stats = run_episode(
            env,
            agent,
            max_timesteps=max_timesteps,
            deterministic=False,
            do_training=True,
            rendering=False,
            skip_frames=2)

        tensorboard.write_episode_data(
            i,
            eval_dict={
                "episode_reward": stats.episode_reward,
                "straight": stats.get_action_usage(STRAIGHT),
                "left": stats.get_action_usage(LEFT),
                "right": stats.get_action_usage(RIGHT),
                "accel": stats.get_action_usage(ACCELERATE),
                "brake": stats.get_action_usage(BRAKE)
            })

        if i % MODEL_TEST_INTERVAL == 0 or i >= (num_episodes - 1):
            # evaluate your agent once in a while for some episodes using run_episode(env, agent, deterministic=True, do_training=False) to
            # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.

            stats_test = run_episode(
                env,
                agent,
                max_timesteps=max_timesteps,
                deterministic=True,
                do_training=False,
                rendering=False,
                skip_frames=2)

            tensorboard_test.write_episode_data(
                i,
                eval_dict={
                    "episode_reward": stats.episode_reward,
                    "straight": stats.get_action_usage(STRAIGHT),
                    "left": stats.get_action_usage(LEFT),
                    "right": stats.get_action_usage(RIGHT),
                    "accel": stats.get_action_usage(ACCELERATE),
                    "brake": stats.get_action_usage(BRAKE)
                })

        # store model every 100 episodes and in the end.
        if i % MODEL_SAVE_INTERVAL == 0 or i >= (num_episodes - 1):
            agent.saver.save(agent.sess,
                             os.path.join(ckpt_dir, "dqn_agent.ckpt"))

    tensorboard.close_session()
    tensorboard_test.close_session()


def make_racecar_agent(name,
                       hidden=20,
                       lr=1e-4,
                       discount_factor=0.99,
                       batch_size=64,
                       epsilon=0.05,
                       epsilon_decay=0.0,
                       boltzmann=False,
                       tau=0.01,
                       double_q=False,
                       buffer_capacity=5e5,
                       save_hypers=True):

    # hidden doesn't do anything here

    hypers = locals()

    num_actions = 5

    # prepare a model folder
    base_path = os.path.join('.', 'racecar')
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    model_path = os.path.join(base_path, name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # save hypers into folder -- used for reconstructing model at test time
    if save_hypers:
        with open(os.path.join(model_path, "hypers.json"), "w") as fh:
            json.dump(hypers, fh)

    # using -1 for unused parameters. fix later.
    Q_current = CNN(state_dim=-1, num_actions=5, hidden=-1, lr=lr)
    Q_target = CNNTargetNetwork(
        state_dim=-1, num_actions=5, hidden=-1, lr=lr, tau=tau)

    # 2. init DQNAgent (see dqn/dqn_agent.py)
    agent = DQNAgent(
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
        random_probs=[3 / 10, 1 / 10, 1 / 10, 3 / 10, 2 / 10])

    return agent, model_path


if __name__ == "__main__":

    env = gym.make('CarRacing-v0').unwrapped

    num_eps = 1000

    agent, model_path = make_racecar_agent('1_basic')
    train_online(env, agent, num_episodes=num_eps, model_dir=model_path)
    
    agent, model_path = make_racecar_agent('2_epsdecay', epsilon_decay=3.33e-4)
    train_online(env, agent, num_episodes=num_eps, model_dir=model_path)

    agent, model_path = make_racecar_agent('3_boltzmann', epsilon=0.0, boltzmann=True)
    train_online(env, agent, num_episodes=num_eps, model_dir=model_path)

    agent, model_path = make_racecar_agent('4_doubleq', double_q=True)
    train_online(env, agent, num_episodes=num_eps, model_dir=model_path)

    env.close()