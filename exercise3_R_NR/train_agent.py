#%%
from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import json

# from model import Model
import kerosey
import tensorflow as tf
from kerosey import Kerosey
from utils import *
from tensorboard_evaluation import Evaluation

#%%


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
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

    # split data into training and validation set
    n_samples = X.shape[0]
    X_train, y_train = X[:int((1 - frac) * n_samples)], y[:int((1 - frac) *
                                                               n_samples)]
    X_valid, y_valid = X[int((1 - frac) * n_samples):], y[int((1 - frac) *
                                                              n_samples):]
    return X_train, y_train, X_valid, y_valid


def process_images(x):
    # add an axis on the end for compulsory depth axis
    return rgb2gray(x)[..., np.newaxis]


def process_actions(y):
    left = y[:, 0] < 0
    right = y[:, 0] > 0
    accel = y[:, 1] > 0
    brake = y[:, 2] > 0
    nothing = np.all(y == 0, axis=1)
    return np.stack((nothing, left, right, accel, brake), axis=1)


def add_history(x, length):
    """adds a dimension to the end of the array, indicating historical offset"""

    if length == 0:
        return x

    x_stack = [x]
    for i in range(length):
        # add a historical shift: episode N of the last X on the stack
        # lines up with episode N-1 of the next X
        x_stack.append(np.roll(x_stack[-1], 1, axis=0))
        # copy starting point so that we don't look backward into the future
        x_stack[-1][0, ...] = x_stack[-2][0, ...]

    return np.concatenate(x_stack, axis=-1)


def augment_training_data(x, y):
    x_flip = np.flip(x, axis=1)  #flip the display horizontally
    y_flip = y[:, [0, 2, 1, 3, 4]]  # reverse steering

    # our balance of acceleration and steering looks fine, so leave it as is

    return np.concatenate((x, x_flip), axis=0), np.concatenate((y, y_flip),
                                                               axis=0)


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)

    X_train, X_valid = [process_images(x) for x in [X_train, X_valid]]

    ### FOR TESTING
    # X_train, y_train, X_valid, y_valid = [
        # x[:1000] for x in [X_train, y_train, X_valid, y_valid]
    # ]

    y_train, y_valid = [process_actions(y) for y in [y_train, y_valid]]

    return X_train, y_train, X_valid, y_valid


def create_a_model(X_shape, y_shape, use_3d, conv_repetitions, num_filters,
                   num_flat_units, drop_prob):

    model = Kerosey.LayeredModel()
    model.setup_input(X_shape, y_shape)

    # for the first convolution, we can either treat history as image depth (independent weights)
    # or part of a 3d image (shared weights). The latter SHOULD work better.

    # 96
    if use_3d:
        model.add_layer(
            Kerosey.Conv3D, filter_size=3, stride=1)

    for rep in range(conv_repetitions):
        model.add_layer(
            Kerosey.Conv2D, filter_size=3, num_filters=num_filters, stride=1)
        model.add_layer(Kerosey.Relu)
        model.add_layer(Kerosey.MaxPool, pool_size=2)

    model.add_layer(Kerosey.Flatten)
    model.add_layer(Kerosey.Dense, num_units=num_flat_units)
    model.add_layer(Kerosey.Relu)

    if drop_prob > 0:
        model.add_layer(Kerosey.Dropout, drop_prob=drop_prob)

    model.add_layer(Kerosey.Dense, num_units=y_shape[1])
    model.compile(loss='crossentropy')

    return model


def train_a_model(name, do_augmentation, history_length, use_3d,
                  conv_repetitions, num_filters, num_flat_units, drop_prob,
                  optimiser, learning_rate, epochs, batch_size):
    """
    This creates and trains a model according to some dataset and sticks it into a folder
    along with CNN training results
    """
    print("MODEL: " + name)

    hypers = locals()

    model_path = os.path.join('.', 'models', name)

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    print("\t... recording hyperparameters")
    with open(os.path.join(model_path, "cnn_hypers.json"), "w") as fh:
        json.dump(hypers, fh)

    # augment training data
    if do_augmentation:
        print("\t... augmenting training data")
        X_train_a, y_train_a = augment_training_data(X_train, y_train)
    else:
        X_train_a, y_train_a = X_train, y_train

    # add history dimension to X data (in image channels, for now)
    print("\t... adding history")
    X_train_ah, X_valid_h = [
        add_history(x, history_length) for x in [X_train_a, X_valid]
    ]

    print("\t... creating model " + str(X_train_ah.shape) + "->" + str(y_train_a.shape))
    model = create_a_model(
        X_shape=X_train_ah.shape,
        y_shape=y_train_a.shape,
        use_3d=use_3d,
        conv_repetitions=conv_repetitions,
        num_filters=num_filters,
        num_flat_units=num_flat_units,
        drop_prob=drop_prob)

    print("\t... training model")
    model.train(
        x_train=X_train_ah,
        y_train=y_train_a,
        x_valid=X_valid_h,
        y_valid=y_valid,
        optimiser=optimiser,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size)

    print("\t... writing training results")
    results = dict()
    results["train_losses"] = model.train_losses
    results["train_accs"] = model.train_accs
    results["valid_losses"] = model.valid_losses
    results["valid_accs"] = model.valid_accs

    with open(os.path.join(model_path, "cnn_results.json"), "w") as fh:
        json.dump(results, fh)

    print("\t... saving checkpoint data")
    model.save(os.path.join(model_path, "ckpt"))

    pass


#%%
import importlib
import kerosey
importlib.reload(kerosey)
from kerosey import Kerosey
# %%

if __name__ == "__main__":
    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, history_length=1)

    #yep
    train_a_model(
        "test_3d",
        do_augmentation=False,
        history_length=5,
        use_3d=True,
        conv_repetitions=2,
        num_filters=16,
        num_flat_units=128,
        drop_prob=0,
        optimiser='adam',
        learning_rate=1e-5,
        epochs=5,
        batch_size=64)
