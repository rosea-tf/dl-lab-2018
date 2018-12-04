#%% Imports

from __future__ import print_function

import pickle
import gzip
import numpy as np
import os
import gc

# %% Functions

def read_data(datasets_dir="./drive_manually"):
    """
    This method reads the states and actions recorded in drive_manually.py 
    """

    X_chunks = []
    y_chunks = []

    for data_file in os.listdir(datasets_dir):
        if data_file.endswith(".pkl.gzip"):
            print("... reading driving data from " + data_file)

            f = gzip.open(os.path.join(datasets_dir, data_file), 'rb')
            data = pickle.load(f)

            # get images as features and actions as targets
            X_chunks.append(np.array(data["state"]).astype('float32'))
            y_chunks.append(np.array(data["action"]).astype('float32'))

    X = np.vstack(X_chunks)
    y = np.vstack(y_chunks)

    return X, y


def process_images(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])
    gray = 2 * gray.astype('float32') - 1

    # add an axis on the end for compulsory depth
    gray = gray[..., np.newaxis]

    return gray


def process_actions(y):
    """discretises actions, regardless of numerical value"""
    left = y[:, 0] < 0
    right = y[:, 0] > 0
    accel = np.logical_and(y[:, 1] > 0, y[:, 0] == 0)
    brake = np.logical_and(y[:, 2] > 0, y[:, 0] == 0)
    nothing = np.all(y == 0, axis=1)
    stacked = np.stack((nothing, left, right, accel, brake), axis=1)

    # if we have multiple actions, do nothing.
    cleaned = np.where(
        stacked.sum(axis=1, keepdims=True) == 1, stacked,
        np.array([1, 0, 0, 0, 0]))  #this should get broadcast

    return cleaned


def add_history(x, length):
    """adds a dimension to the end of the array, indicating historical offset"""

    if length == 0:
        return x

    x_stack = np.empty(x.shape[:-1] + (length, ))
    x_stack[..., 0] = x[..., 0]
    
    for i in range(1, length):
        # add a historical shift: episode N of the last X on the stack
        # lines up with episode N-1 of the next X
        print("\t\t ... ", i)

        x_stack[1:, ..., i] = x_stack[0:-1, ..., i - 1]

        # copy starting point so that we don't look backward into the future
        x_stack[0, ..., i] = x_stack[0, ..., i - 1]

    return x_stack

def rebalance_actions(x, y):
    """returns roughly the same size of dataset, but with balanced y-actions"""

    y_action_probs = y.mean(axis=0)
    y_actions = np.argmax(y, axis=1)
    y_action_sample_probs = np.min(y_action_probs) / y_action_probs
    y_sample_probs = y_action_sample_probs[
        y_actions]  # each row's sample probability

    y_samples_required = np.int(
        np.round(1 / y_sample_probs.mean(), decimals=0))

    x_stack = []
    y_stack = []

    for i in range(y_samples_required):
        y_sample = np.random.binomial(1, (y_sample_probs))
        x_stack.append(x[y_sample == 1])
        y_stack.append(y[y_sample == 1])

    x_resampled = np.concatenate(x_stack, axis=0)
    y_resampled = np.concatenate(y_stack, axis=0)

    return x_resampled, y_resampled


def add_horizontal_flip(x, y):
    """was used for data augmentation by flipping images horizontally and reversing steering
    but it caused memory problems -- we have too much data anyway"""
    return x, y #kill this - memory error.

    x_flip = np.flip(x, axis=2)  #flip the display horizontally (bug fix. oh dear.)
    y_flip = y[:, [0, 2, 1, 3, 4]]  # reverse steering

    x_cat, y_cat = np.concatenate((x, x_flip), axis=0), np.concatenate(
        (y, y_flip), axis=0)

    return x_cat, y_cat


def preprocessing(X, y):

    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    print ("processing images...")
    X = process_images(X)

    print ("discretising actions...")
    y = process_actions(y)

    return X, y

if __name__ == '__main__':

    X, y = read_data("./drive_manually")
    X, y = preprocessing(X, y)

    # this seems useful
    n_samples = X.shape[0]

    history_length = 4
    print ("HISTORY: ", history_length)
    print("\t... adding history")
    X = add_history(X, history_length)

    gc.collect()

    print("\t... shuffling")

    np.random.seed(123)
    np.random.shuffle(X)
    np.random.seed(123)
    np.random.shuffle(y)

    print("\t... splitting")

    # now the t/v split
    frac = 0.1
    X_train, y_train = X[:int((1 - frac) * n_samples)], y[:int((1 - frac) *
                                                               n_samples)]
    X_valid, y_valid = X[int((1 - frac) * n_samples):], y[int((1 - frac) *
                                                              n_samples):]
    gc.collect()

    print("\t... saving data (1)")

    # try a version without rebalancing    
    with gzip.open(os.path.join('datasets', 'xy0nr.pkl.gz'), 'wb') as fh:
        pickle.dump([X_train, y_train, X_valid, y_valid], fh)
        
    print("\t... rebalancing")
    X_train, y_train = rebalance_actions(X_train, y_train)

    print("\t... saving rebalanced data (2)")

    with gzip.open(os.path.join('datasets', 'xy3.pkl.gz'), 'wb') as fh:
        pickle.dump([X_train, y_train, X_valid, y_valid], fh)
