from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json
import argparse

from train_agent import create_a_model, process_images

# %%

IMAGE_W, IMAGE_H = 96, 96
OUTPUT_SIZE = 5
#(nothing, left, right, accel, brake)
ACTION_TYPES = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0], [0.0, 0.0, 0.4]])


def action_from_prediction(p):
    """
    takes a len-5 array from network (nothing, left, right, accel, brake)
    returns a len-3 action expected by gym (steer L-/R+, accel, brake)
    """
    m = np.argmax(
        p
    )  # choose most likely action (not sampling from a softmax here... yet)
    return ACTION_TYPES[m]


def retrieve_model(directory):
    fname = os.path.join("models", directory, "cnn_hypers.json")

    with open(fname, "r") as fh:
        hypers = json.load(fh)

    # create a model of the same spec that was saved in the folder
    # because tensorflow won't do this for you.
    model = create_a_model(
        X_shape=(2000, IMAGE_W, IMAGE_H, 1 + hypers['history_length']),
        y_shape=(2000, OUTPUT_SIZE),
        use_3d=hypers['use_3d'],
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
        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py
        state_g = process_images(state)[np.newaxis, ...]

        if history_length > 0:
            if history is None:
                history = np.broadcast_to(
                    state_g, shape=(1, IMAGE_W, IMAGE_H,
                                    1 + history_length)).copy()
            else:
                history[..., 1:] = history[..., :
                                           -1]  # shift all elements back 1
                history[..., 0] = state_g[
                    ..., 0]  #copy curent image into 0 position
        else:
            history = state_g

        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        prediction = np.squeeze(agent.run(agent.prediction(), history, None))

        action = action_from_prediction(prediction)

        next_state, r, done, info = env.step(action)
        episode_reward += r
        state = next_state

        if step % 100 == 0:
            print('step', step)

        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


#%%
if False:
    # %%
    agent, hypers = retrieve_model('1a_small')

# %%

# %%
if __name__ == "__main__":

    #%%
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", help="directory from which to retrieve trained model")
    args = parser.parse_args()

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 15  # number of episodes to test
    # n_test_episodes = 1

    agent, hypers = retrieve_model(args.directory)

    env = gym.make('CarRacing-v0').unwrapped

    # %%
    episode_rewards = []

    for i in range(n_test_episodes):
        # pass hypers because it contains history_length
        episode_reward = run_episode(env, agent, hypers, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    # keep in the same folder as the model
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    fname = os.path.join(
        "models", args.directory,
        "results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
