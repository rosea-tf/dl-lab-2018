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
from test_racecar import evaluate_agent

base_path = os.path.join('.', 'racecar')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        help="directory name under ./racecar/ of trained model to retrieve (or ALL)"
    )
    parser.add_argument(
        "--softmax",
        type=bool,
        default=False,
        help="true or false"
    )
    args = parser.parse_args()

    evaluate_agent(args.name, n_test_episodes=1, rendering=True, softmax=args.softmax)
