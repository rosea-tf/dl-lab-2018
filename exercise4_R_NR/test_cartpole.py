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
    #     "directory", help="directory from which to retrieve trained model")
    # args = parser.parse_args()
    # model_dir = args.directory
    
    model_dir = 'cartpole'

    env = gym.make("CartPole-v0").unwrapped

    agent = make_cartpole_agent()
    
    agent.load(os.path.join(model_dir, 'ckpt', 'dqn_agent.ckpt'))
 
    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    # if not os.path.exists("./results"):
    #     os.mkdir("./results")  

    fname = "./cartpole/results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

