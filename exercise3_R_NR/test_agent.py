from __future__ import print_function

import numpy as np
import gym
import os
import json
import argparse
import gc

from train_agent import create_a_model
from process_data import process_images

# %%

IMAGE_W, IMAGE_H = 96, 96
OUTPUT_SIZE = 5
#(nothing, left, right, accel, brake)
ACTION_TYPES = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0], [0.0, 0.0, 0.4]])

MODELS_DIR = "models"
TEST_RESULTS_FNAME = "test_results_sm.json"

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def action_from_prediction(p):
    """
    takes a len-5 array from network (nothing, left, right, accel, brake)
    returns a len-3 action expected by gym (steer L-/R+, accel, brake)
    """
    
    ps = softmax(p)
    
    a = (ps[2] - ps[1], ps[3], ps[4] * 0.4)

    return a


def retrieve_model(directory):
    fname = os.path.join(MODELS_DIR, directory, "cnn_hypers.json")

    with open(fname, "r") as fh:
        hypers = json.load(fh)

    # create a model of the same spec that was saved in the folder
    # because tensorflow won't do this for you.
    model = create_a_model(
        X_shape=(2000, IMAGE_W, IMAGE_H, hypers['history_length']),
        y_shape=(2000, OUTPUT_SIZE),
        sequentialise=hypers['sequentialise'],
        use_lstm=hypers['use_lstm'],
        filter_size=3,
        num_filters=hypers['num_filters'],
        num_flat_units=hypers['num_flat_units'],
        drop_prob=hypers['drop_prob'],
        optimiser=hypers['optimiser'],
        learning_rate=hypers['learning_rate'])

    # load up its weights or whatever is in here
    model.restore(os.path.join(".", "models", directory, "ckpt"))

    return model, hypers


def run_episode(env, agent, hypers, rendering=True, max_timesteps=1000):

    episode_reward = 0
    step = 0

    state = env.reset()

    history_length = hypers['history_length']
    history = None

    #idea: keep an np.array of last 5 states. expand current one to begin with. then roll, every step.

    while True:
        state_g = process_images(state)[np.newaxis, ...]

        if history_length > 1:
            if history is None:
                history = np.broadcast_to(
                    state_g, shape=(1, IMAGE_W, IMAGE_H,
                                    history_length)).copy()
            else:
                history[..., 1:] = history[..., :
                                           -1]  # shift all elements back 1
                history[..., 0] = state_g[
                    ..., 0]  #copy curent image into 0 position
        else:
            history = state_g

        prediction = np.squeeze(agent.run(agent.prediction(), history, None))

        action = action_from_prediction(prediction)

        next_state, r, done, info = env.step(action)
        episode_reward += r
        state = next_state

        if step % 100 == 0:
            print('step', step)
            #reset agent - fixing tf slowdown
            agent, hypers = retrieve_model(hypers['name'])
            gc.collect()

        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


# %%
def evaluate_agent(directory, n_test_episodes, rendering):

    # %%
    episode_rewards = []

    env = gym.make('CarRacing-v0').unwrapped

    env.seed(1234)

    for i in range(n_test_episodes):
        # pass hypers because it contains history_length
        agent, hypers = retrieve_model(directory)
        # it seems to slow down unbearably unless we refresh the agent every so often

        episode_reward = run_episode(env, agent, hypers, rendering=rendering)
        episode_rewards.append(episode_reward)

        agent = None
        gc.collect()

        # save results in a dictionary and write them into a .json file
        # keep in the same folder as the model
        # do this every episode, in case we need to bail out early
        results = dict()
        results["episode_rewards"] = episode_rewards
        results["mean"] = np.array(episode_rewards).mean()
        results["std"] = np.array(episode_rewards).std()

        fname = os.path.join(MODELS_DIR, directory, TEST_RESULTS_FNAME)
        with open(fname, "w") as fh:
            json.dump(results, fh)

    env.close()

    env = None

    print('... finished')

    return


# %%
if __name__ == "__main__":

    #%%
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", help="directory from which to retrieve trained model")
    args = parser.parse_args()

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 10  # number of episodes to test

    if args.directory == 'ALL':
        for directory in os.listdir(MODELS_DIR):
            dirpath = os.path.join(MODELS_DIR, directory)
            if os.path.isdir(dirpath):
                if os.path.exists(os.path.join(dirpath, 'cnn_results.json')):
                    print("Found model at " + dirpath)
                    # don't run test if a results file already exists
                    if not os.path.exists(
                            os.path.join(dirpath, TEST_RESULTS_FNAME)):
                        evaluate_agent(directory, n_test_episodes, rendering)
                    else:
                        print("\t...already tested! skipping")

    else:
        evaluate_agent(args.directory, n_test_episodes, rendering)
