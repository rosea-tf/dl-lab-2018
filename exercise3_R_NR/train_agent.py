#%% Imports

from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import json

from kerosey import Kerosey

# %% Delete this later

import importlib
import kerosey
importlib.reload(kerosey)

#%% Function Definitions


def read_data(datasets_dir="./drive_manually", frac=0.1, toy=False):
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

            if toy:
                X_chunks[0], y_chunks[0] = [
                    x[:1000] for x in [X_chunks[0], y_chunks[0]]
                ]

                break

    X = np.vstack(X_chunks)
    y = np.vstack(y_chunks)

    # split data into training and validation set
    n_samples = X.shape[0]
    X_train, y_train = X[:int((1 - frac) * n_samples)], y[:int((1 - frac) *
                                                               n_samples)]
    X_valid, y_valid = X[int((1 - frac) * n_samples):], y[int((1 - frac) *
                                                              n_samples):]
    return X_train, y_train, X_valid, y_valid


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
    accel = y[:, 1] > 0
    brake = y[:, 2] > 0
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

    x_stack = [x]
    for i in range(length):
        # add a historical shift: episode N of the last X on the stack
        # lines up with episode N-1 of the next X
        x_stack.append(np.roll(x_stack[-1], 1, axis=0))
        # copy starting point so that we don't look backward into the future
        x_stack[-1][0, ...] = x_stack[-2][0, ...]

    return np.concatenate(x_stack, axis=-1)


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
    x_flip = np.flip(x, axis=1)  #flip the display horizontally
    y_flip = y[:, [0, 2, 1, 3, 4]]  # reverse steering

    # our balance of acceleration and steering looks fine, so leave it as is

    x_cat, y_cat = np.concatenate((x, x_flip), axis=0), np.concatenate(
        (y, y_flip), axis=0)

    return x_cat, y_cat


def preprocessing(X_train, y_train, X_valid, y_valid):

    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    X_train, X_valid = [process_images(x) for x in [X_train, X_valid]]

    # 2. discretise actions
    y_train, y_valid = [process_actions(y) for y in [y_train, y_valid]]

    return X_train, y_train, X_valid, y_valid


def create_a_model(X_shape, y_shape, use_3d, num_filters, num_flat_units,
                   drop_prob, optimiser, learning_rate):

    model = Kerosey.LayeredModel()
    model.setup_input(X_shape, y_shape)

    # for the first convolution, we can either treat history as image depth (independent weights)
    # or part of a 3d image (shared weights). The latter SHOULD work better.

    if use_3d:
        model.add_layer(Kerosey.Sequentialise)

    for rep, nfs in enumerate(num_filters):

        if use_3d:
            model.add_layer(
                Kerosey.ConvSeq2D,
                filter_size=3,
                num_filters=nfs,
                stride=1,
                padding='VALID')
        else:
            model.add_layer(
                Kerosey.Conv2D,
                filter_size=3,
                num_filters=nfs,
                stride=1,
                padding='VALID')

        model.add_layer(Kerosey.Relu)

        if use_3d:
            model.add_layer(Kerosey.MaxPoolSeq, pool_size=2)
        else:
            model.add_layer(Kerosey.MaxPool, pool_size=2)

    if use_3d:
        # model.add_layer(Kerosey.FlattenSeq) # this worked!
        model.add_layer(Kerosey.FlattenSeq, keep_seq=True)
        model.add_layer(Kerosey.LSTM, num_units=(num_flat_units[0]*4)) #?
    else:
        model.add_layer(Kerosey.Flatten)

    if drop_prob > 0:
        model.add_layer(Kerosey.Dropout, drop_prob=drop_prob)

    for num_flats in num_flat_units:
        model.add_layer(Kerosey.Dense, num_units=num_flats)
        model.add_layer(Kerosey.Relu)

    # two dropout points is probably enough
    if drop_prob > 0:
        model.add_layer(Kerosey.Dropout, drop_prob=drop_prob)

    model.add_layer(Kerosey.Dense, num_units=y_shape[1])
    model.compile(
        loss='crossentropy', optimiser=optimiser, learning_rate=learning_rate)

    return model


def train_a_model(name, do_augmentation, history_length, use_3d, num_filters,
                  num_flat_units, drop_prob, optimiser, learning_rate, epochs,
                  batch_size):
    """
    This creates and trains a model according to some dataset and sticks it into a folder
    along with CNN training results
    """
    print("MODEL: " + name)

    hypers = locals()

    model_path = os.path.join('.', 'models', name)

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if os.path.exists(os.path.join(model_path, "cnn_results.json")):
        # model already trained: skip
        pass

    print("\t... recording hyperparameters")
    with open(os.path.join(model_path, "cnn_hypers.json"), "w") as fh:
        json.dump(hypers, fh)

    # add history dimension to X data (in image channels, for now)
    print("\t... adding history")
    X_train_h, X_valid_h = [
        add_history(x, history_length) for x in [X_train, X_valid]
    ]

    # augment training data
    if do_augmentation:
        print("\t... rebalancing and augmenting")
        X_train_ha, y_train_a = add_horizontal_flip(
            *rebalance_actions(X_train_h, y_train))
    else:
        X_train_ha, y_train_a = X_train_h, y_train

    # this seems useful for minibatching
    p = np.random.permutation(X_train_ha.shape[0])
    X_train_ha = X_train_ha[p]
    y_train_a = y_train_a[p]

    print("\t... creating model " + str(X_train_ha.shape) + "->" +
          str(y_train_a.shape))
    model = create_a_model(
        X_shape=X_train_ha.shape,
        y_shape=y_train_a.shape,
        use_3d=use_3d,
        num_filters=num_filters,
        num_flat_units=num_flat_units,
        drop_prob=drop_prob,
        optimiser=optimiser,
        learning_rate=learning_rate)

    print("\t... training model")
    model.train(
        x_train=X_train_ha,
        y_train=y_train_a,
        x_valid=X_valid_h,
        y_valid=y_valid,
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


#%% Programme

if __name__ == "__main__":
    # read data
    # X_train, y_train, X_valid, y_valid = read_data("./drive_manually")
    X_train, y_train, X_valid, y_valid = read_data(
        "./drive_manually", 0.1, toy=True)

    quit

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train,
                                                       X_valid, y_valid)

    #yep
    # train_a_model(
    #     "1_basic",
    #     do_augmentation=False,
    #     history_length=0,
    #     use_3d=False,
    #     num_filters=[16, 16],
    #     num_flat_units=[512, 128],
    #     drop_prob=0,
    #     optimiser='adam',
    #     learning_rate=1e-5,
    #     epochs=20,
    #     batch_size=32)

    # train_a_model(
    #     "2_augmented",
    #     do_augmentation=True,
    #     history_length=0,
    #     use_3d=False,
    #     num_filters=[16, 16],
    #     num_flat_units=[512, 128],
    #     drop_prob=0,
    #     optimiser='adam',
    #     learning_rate=1e-5,
    #     epochs=20,
    #     batch_size=32)

    # train_a_model(
    #     "3a_dropout05",
    #     do_augmentation=True,
    #     history_length=0,
    #     use_3d=False,
    #     num_filters=[16, 16],
    #     num_flat_units=[512, 128],
    #     drop_prob=0.05,
    #     optimiser='adam',
    #     learning_rate=1e-5,
    #     epochs=20,
    #     batch_size=32)

    # train_a_model(
    #     "3b_dropout10",
    #     do_augmentation=True,
    #     history_length=0,
    #     use_3d=False,
    #     num_filters=[16, 16],
    #     num_flat_units=[512, 128],
    #     drop_prob=0.1,
    #     optimiser='adam',
    #     learning_rate=1e-5,
    #     epochs=20,
    #     batch_size=32)

    # train_a_model(
    #     "3c_dropout20",
    #     do_augmentation=True,
    #     history_length=0,
    #     use_3d=False,
    #     num_filters=[16, 16],
    #     num_flat_units=[512, 128],
    #     drop_prob=0.2,
    #     optimiser='adam',
    #     learning_rate=1e-5,
    #     epochs=20,
    #     batch_size=32)

    #currently running on 60

    # train_a_model(
    #     "4_larger50",
    #     do_augmentation=True,
    #     history_length=0,
    #     use_3d=False,
    #     num_filters=[24, 24],
    #     num_flat_units=[768, 128],
    #     drop_prob=0.2,                      #CHECK THIS
    #     optimiser='adam',
    #     learning_rate=1e-5,
    #     epochs=20,
    #     batch_size=32)

    # history 1, 3, 5

    # 3d history

    # lstm??

train_a_model(
    "test",
    do_augmentation=False,
    history_length=2,
    use_3d=True,
    num_filters=[5, 5],
    num_flat_units=[256, 128],
    drop_prob=0,
    optimiser='adam',
    learning_rate=1e-5,
    epochs=3,
    batch_size=32)
