from __future__ import print_function

import gym
from train_carracing import run_episode
from dqn.dqn_agent_car_racing import DQNAgent
from dqn.conv_networks import CNN, CNNTargetNetwork
import numpy as np
import os

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped

    history_length =  0

    #TODO: Define networks and load agent
    q = CNN(env.observation_space.shape[0], 5)
    q_target = CNNTargetNetwork(96*96, 5)

    agent = DQNAgent(q, q_target, 5)
    agent.load(os.path.join("models_carracing", "dqn_agent.ckpt"))

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

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')

