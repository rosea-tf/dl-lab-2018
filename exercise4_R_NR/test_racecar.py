import os
from datetime import datetime
import gym
import json
from dqn.dqn_agent import DQNAgent
from train_racecar import run_episode, make_racecar_agent
from dqn.conv_networks import CNN, CNNTargetNetwork
import numpy as np
import argparse
import tensorflow as tf

base_path = os.path.join('.', 'racecar')
n_test_episodes = 15
rendering = True

np.random.seed(0)


def evaluate_agent(model_name, softmax=False):
    print ("MODEL: " + model_name + "" if not softmax else " (softmax))")

    tf.reset_default_graph() #this seems to help consecutive testing
    model_path = os.path.join(base_path, model_name)

    # check if already done
    results_fn = "results.json" if not softmax else "results_sm.json"
    
    if os.path.exists(os.path.join(model_path, results_fn)):
        print ("\t... results exist already! skipping.")
        return

    # get hypers from the model in this folder
    with open(os.path.join(model_path, "hypers.json"), "r") as fh:
        hypers = json.load(fh)

    #backwards compatibility
    if 'history_length' not in hypers.keys():
        hypers['history_length'] = 0
    if 'diff_history' not in hypers.keys():
        hypers['diff_history'] = False

    env = gym.make("CarRacing-v0").unwrapped

    # some of these hypers won't matter once training is over, but anyway...

    if not softmax:
        agent = make_racecar_agent(
            name=model_name,
            model_path=model_path,
            lr=hypers['lr'],
            discount_factor=hypers['discount_factor'],
            batch_size=hypers['batch_size'],
            epsilon=hypers['epsilon'],
            epsilon_decay=hypers['epsilon_decay'],
            boltzmann=hypers['boltzmann'],
            tau=hypers['tau'],
            double_q=hypers['double_q'],
            buffer_capacity=hypers['buffer_capacity'],
            history_length=hypers['history_length'],
            diff_history=hypers['diff_history'],
            save_hypers=False)
    else:
            agent = make_racecar_agent(
            name=model_name,
            model_path=model_path,
            lr=hypers['lr'],
            discount_factor=hypers['discount_factor'],
            batch_size=hypers['batch_size'],
            epsilon=0.0,
            epsilon_decay=0.0,
            boltzmann=True,
            tau=hypers['tau'],
            double_q=hypers['double_q'],
            buffer_capacity=hypers['buffer_capacity'],
            history_length=hypers['history_length'],
            diff_history=hypers['diff_history'],
            save_hypers=False)

    # retrieve weights
    agent.load(os.path.join(model_path, 'ckpt', 'dqn_agent.ckpt'))

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(
            env,
            agent,
            deterministic=(not softmax),
            do_training=False,
            rendering=rendering,
            history_length=hypers['history_length'],
            diff_history=hypers['diff_history'])
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    with open(os.path.join(model_path, results_fn), "w") as fh:
        json.dump(results, fh)

    env.close()
    print('... finished')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        help=
        "(directory name under ./racecar/ of trained model to retrieve (or ALL)"
    )
    args = parser.parse_args()

    if args.name == 'ALL':
        for thing in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, thing)):
                if os.path.exists(os.path.join(base_path, thing, "hypers.json")):
                    evaluate_agent(thing)
                    evaluate_agent(thing, softmax=True)

    else:
        evaluate_agent(args.name)
        evaluate_agent(args.name, softmax=True)
