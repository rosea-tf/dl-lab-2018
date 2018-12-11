import os
from datetime import datetime
import gym
import json
from dqn.dqn_agent import DQNAgent
from train_cartpole import run_episode, make_cartpole_agent
from dqn.networks import *
import numpy as np
import argparse

np.random.seed(0)

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "name", help="(directory name under ./cartpole/ of trained model to retrive")
    # args = parser.parse_args()
    # model_name = args.name

    model_name = 'basic'  #TODO uncomment

    base_path = os.path.join('.', 'cartpole')
    model_path = os.path.join(base_path, model_name)

    # get hypers from the model in this folder
    with open(os.path.join(model_path, "hypers.json"), "r") as fh:
        hypers = json.load(fh)

    env = gym.make("CartPole-v0").unwrapped

    # some of these hypers won't matter once training is over, but anyway...
    agent, _ = make_cartpole_agent(
        name=model_name,
        hidden=hypers['hidden'],
        lr=hypers['lr'],
        discount_factor=hypers['discount_factor'],
        batch_size=hypers['batch_size'],
        epsilon=hypers['epsilon'],
        tau=hypers['tau'],
        double_q=hypers['double_q'],
        eps_method=hypers['eps_method'],
        save_hypers=False)

    # retrieve weights
    agent.load(os.path.join(model_path, 'ckpt', 'dqn_agent.ckpt'))

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(
            env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()


    with open(os.path.join(model_path, "results.json"), "w") as fh:
        json.dump(results, fh)

    env.close()
    print('... finished')
