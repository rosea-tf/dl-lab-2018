import numpy as np
import gym
from dqn.dqn_agent import DQNAgent
from tensorboard_evaluation import Evaluation
from dqn.conv_networks import CNN, CNNTargetNetwork
from utils import *
import os
import json

MODEL_TEST_INTERVAL = 10  # after this number of episodes, test agent with deterministic actions
MODEL_SAVE_INTERVAL = 100  # yep


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

def is_close(a,colors,eps=0.02):
    for color in colors:
        if abs(a-color) < eps:
            return True

    return False


def lane_penalty(state):
    green_colors = [0.68, 0.75]

    ## check for green values (174.??? or 192.???)
    left_sensor = state[68, 44, 0]
    right_sensor = state[68, 51, 0]

    offlane_left = is_close(left_sensor, green_colors)
    offlane_right = is_close(right_sensor, green_colors)

    if offlane_left and offlane_right:
        return -1.0  # bad buggy: completely of track, high penalty
    elif offlane_left or offlane_right:
        return -0.0  # one side off track
    else:
        return 0.0


def run_episode(env,
                agent,
                deterministic,
                softmax=False,
                skip_frames=0,
                do_training=True,
                rendering=False,
                max_timesteps=1000,
                history_length=0,
                diff_history=False,
                apply_lane_penalty=False):
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
    # state = np.array(image_hist).reshape(96, 96, history_length + 1)
    state = np.array(image_hist[::-1]).transpose(1, 2, 0)

    if diff_history:
        # current image is at zero, so go through 1...n and subtract current from them
        for i in range(1, state.shape[-1]):
            state[..., i] -= state[..., 0]

    while True:

        action_id = agent.act(state=state, deterministic=deterministic)
        action = id_to_action(action_id)

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
        # next_state = np.array(image_hist).reshape(96, 96, history_length + 1)
        next_state = np.array(image_hist[::-1]).transpose(1, 2, 0)

        # so state and next_state are now both np.arrays, of the right length
        
        if diff_history:
            # current image is at zero, so go through 1...n and subtract current from them
            for i in range(1, next_state.shape[-1]):
                next_state[..., i] -= next_state[..., 0]

        # only after zooming (after 50 steps), apply lane penalty
        if step > 50/(skip_frames+1) and apply_lane_penalty:
            reward += lane_penalty(next_state) # add lane penalty to reward

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    return stats


def train_online(name,
                 env,
                 num_episodes=1000,
                 lr=1e-4,
                 discount_factor=0.99,
                 batch_size=64,
                 epsilon=0.05,
                 epsilon_decay=0.0,
                 boltzmann=False,
                 tau=0.01,
                 double_q=False,
                 buffer_capacity=5e5,
                 history_length=0,
                 diff_history=False,
                 apply_lane_penalty=False,
                 try_resume=False):

    print("AGENT: " + name)
    print("\t... creating agent")

    # prepare folders
    model_path = os.path.join(base_path, name)
    ckpt_path = os.path.join(model_path, "ckpt")
    tensorboard_path = os.path.join(base_path, "tensorboard")

    for d in [model_path, ckpt_path, tensorboard_path]:
        if not os.path.exists(d):
            os.mkdir(d)

    agent = make_racecar_agent(
        name,
        model_path,
        lr,
        discount_factor,
        batch_size,
        epsilon,
        epsilon_decay,
        boltzmann,
        tau,
        double_q,
        buffer_capacity,
        history_length,
        diff_history,
        save_hypers=True)

    print("... training agent")

    # todo? make this better
    tensorboard = Evaluation(
        os.path.join(tensorboard_path, agent.name + "_train"),
        ["episode_reward", "straight", "left", "right", "accel", "brake"])
    tensorboard_test = Evaluation(
        os.path.join(tensorboard_path, agent.name + "_test"),
        ["episode_reward", "straight", "left", "right", "accel", "brake"])

    start_episode = 0

    if try_resume:
        possible_file = os.path.join(model_path, "epstrained.json")
        if os.path.exists(possible_file):
            # get the last ep trained; start at the next one
            with open(possible_file, "r") as fh:
                start_episode = json.load(fh) + 1

            #load up model from previous training session
            agent.load(os.path.join(model_path, 'ckpt', 'dqn_agent.ckpt'))

    # training
    for i in range(start_episode, num_episodes):
        print("episode: ", i)

        max_timesteps = min(
            1000, 4 * i + 100)  #adr - sped up compared to ingmar's version

        stats = run_episode(
            env,
            agent,
            max_timesteps=max_timesteps,
            deterministic=False,
            softmax=False,
            do_training=True,
            rendering=False,
            skip_frames=2,
            history_length=history_length,
            diff_history=diff_history,
            apply_lane_penalty=apply_lane_penalty)

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

            stats_test = run_episode(
                env,
                agent,
                max_timesteps=max_timesteps,
                deterministic=True,
                softmax=False,
                do_training=False,
                rendering=False,
                skip_frames=2,
                history_length=history_length,
                diff_history=diff_history)

            tensorboard_test.write_episode_data(
                i,
                eval_dict={
                    "episode_reward": stats_test.episode_reward,
                    "straight": stats_test.get_action_usage(STRAIGHT),
                    "left": stats_test.get_action_usage(LEFT),
                    "right": stats_test.get_action_usage(RIGHT),
                    "accel": stats_test.get_action_usage(ACCELERATE),
                    "brake": stats_test.get_action_usage(BRAKE)
                })

        # store model every 100 episodes and in the end.
        if i % MODEL_SAVE_INTERVAL == 0 or i >= (num_episodes - 1):
            agent.saver.save(agent.sess,
                             os.path.join(ckpt_path, "dqn_agent.ckpt"))

        # write an episode counter, so that we can resume training later
        with open(os.path.join(model_path, "epstrained.json"), "w") as fh:
            json.dump(i, fh)

    tensorboard.close_session()
    tensorboard_test.close_session()
    
    # run the testing from here
    os.system("python3 test_racecar.py " + name)


def make_racecar_agent(name, model_path, lr, discount_factor,
                       batch_size, epsilon, epsilon_decay, boltzmann, tau,
                       double_q, buffer_capacity, history_length, diff_history,
                       save_hypers):

    hypers = locals()

    num_actions = 5

    # save hypers into folder -- used for reconstructing model at test time
    if save_hypers:
        with open(os.path.join(model_path, "hypers.json"), "w") as fh:
            json.dump(hypers, fh)

    # using -1 for unused parameters. fix later.
    Q_current = CNN(
        num_actions=5,
        lr=lr,
        history_length=history_length,
        diff_history=diff_history)

    Q_target = CNNTargetNetwork(
        num_actions=5,
        lr=lr,
        tau=tau,
        history_length=history_length,
        diff_history=diff_history)

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

    return agent


if __name__ == "__main__":

    env = gym.make('CarRacing-v0').unwrapped

    # prepare a model folder
    base_path = os.path.join('.', 'racecar')
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    # train_online('1_basic', env)

    # train_online('2_epsdecay', env, epsilon_decay=1e-3)

    # train_online('3_boltzmann', env, epsilon=0.0, boltzmann=True, try_resume=True)

    #train_online('4_doubleq', env, double_q=True)

    # train_online('5_nodiscount', env, discount_factor=1.0)

    # train_online('6_negdiscount', env, discount_factor=1.01)

    #train_online('7_history', env, history_length=1, try_resume=True)

    #train_online('8_difframe', env, history_length=1, diff_history=True, try_resume=True)

    train_online('9_diffpenalty', env, history_length=1, diff_history=True, apply_lane_penalty=True, epsilon_decay=1e-3, try_resume=True)

    env.close()
