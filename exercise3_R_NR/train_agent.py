#%% Imports

from __future__ import print_function

import pickle
import os
import gzip
import json

from kerosey import Kerosey

#%% Function Definitions


def create_a_model(X_shape, y_shape, sequentialise, use_lstm, filter_size,
                   num_filters, num_flat_units, drop_prob, optimiser,
                   learning_rate):

    assert (not sequentialise
            ) or X_shape[-1] > 0, "sequentalising requires history"
    assert (not use_lstm) or sequentialise, "LSTM requires sequentialising"

    model = Kerosey.LayeredModel()
    model.setup_input(X_shape, y_shape)

    # for the first convolution, we can either treat history as image depth (independent weights)
    # or part of a 3d image (shared weights). The latter SHOULD work better.

    if sequentialise:
        model.add_layer(Kerosey.Sequentialise)

    for nfs in num_filters:

        if sequentialise:
            model.add_layer(
                Kerosey.ConvSeq2D,
                filter_size=filter_size,
                num_filters=nfs,
                stride=1,
                padding='VALID')
        else:
            model.add_layer(
                Kerosey.Conv2D,
                filter_size=filter_size,
                num_filters=nfs,
                stride=1,
                padding='VALID')

        model.add_layer(Kerosey.Relu)

        if sequentialise:
            model.add_layer(Kerosey.MaxPoolSeq, pool_size=2)
        else:
            model.add_layer(Kerosey.MaxPool, pool_size=2)

    if sequentialise:
        model.add_layer(Kerosey.FlattenSeq, keep_seq=use_lstm)
        if use_lstm:
            model.add_layer(Kerosey.LSTM, num_units=(num_flat_units[0]))  #?
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


def train_a_model(name, do_augmentation, history_length, sequentialise,
                  use_lstm, filter_size, num_filters, num_flat_units,
                  drop_prob, optimiser, learning_rate, epochs, batch_size):
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
        # model already trained: skip?
        print('already exists!')
        return

    print("\t... loading data")

    # read data
    filename = 'xy3.pkl.gz' if do_augmentation else 'xy0nr.pkl.gz'

    with gzip.open(os.path.join('datasets', filename), 'rb') as fh:
        X_train, y_train, X_valid, y_valid = pickle.load(fh)

    if history_length < 4:
        X_train = X_train[..., :history_length]
        X_valid = X_valid[..., :history_length]

    print("\t... creating model " + str(X_train.shape) + "->" +
          str(y_train.shape))
    model = create_a_model(
        X_shape=X_train.shape,
        y_shape=y_train.shape,
        sequentialise=sequentialise,
        use_lstm=use_lstm,
        filter_size=filter_size,
        num_filters=num_filters,
        num_flat_units=num_flat_units,
        drop_prob=drop_prob,
        optimiser=optimiser,
        learning_rate=learning_rate)

    print("\t... training model")
    model.train(
        x_train=X_train,
        y_train=y_train,
        x_valid=X_valid,
        y_valid=y_valid,
        epochs=epochs,
        batch_size=batch_size)

    print("\t... recording hyperparameters")
    with open(os.path.join(model_path, "cnn_hypers.json"), "w") as fh:
        json.dump(hypers, fh)

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


#%% Programme

if __name__ == "__main__":

    train_a_model(
        "1_basic",
        do_augmentation=False,
        history_length=1,
        sequentialise=False,
        use_lstm=False,
        filter_size=3,
        num_filters=[12, 12],
        num_flat_units=[256, 128],
        drop_prob=0,
        optimiser='adam',
        learning_rate=1e-4,
        epochs=20,
        batch_size=32)

    train_a_model(
        "2_augmented",
        do_augmentation=True,
        history_length=1,
        sequentialise=False,
        use_lstm=False,
        filter_size=3,
        num_filters=[12, 12],
        num_flat_units=[256, 128],
        drop_prob=0,
        optimiser='adam',
        learning_rate=1e-4,
        epochs=20,
        batch_size=32)

    train_a_model(
        "3a_dropout05",
        do_augmentation=True,
        history_length=1,
        sequentialise=False,
        use_lstm=False,
        filter_size=3,
        num_filters=[12, 12],
        num_flat_units=[256, 128],
        drop_prob=0.05,
        optimiser='adam',
        learning_rate=1e-4,
        epochs=20,
        batch_size=32)

    train_a_model(
        "3b_dropout10",
        do_augmentation=True,
        history_length=1,
        sequentialise=False,
        use_lstm=False,
        filter_size=3,
        num_filters=[12, 12],
        num_flat_units=[256, 128],
        drop_prob=0.1,
        optimiser='adam',
        learning_rate=1e-4,
        epochs=20,
        batch_size=32)

    train_a_model(
        "3c_dropout20",
        do_augmentation=True,
        history_length=1,
        sequentialise=False,
        use_lstm=False,
        filter_size=3,
        num_filters=[12, 12],
        num_flat_units=[256, 128],
        drop_prob=0.2,
        optimiser='adam',
        learning_rate=1e-4,
        epochs=20,
        batch_size=32)

    train_a_model(
        "4a_hist2",
        do_augmentation=True,
        history_length=2,
        sequentialise=False,
        use_lstm=False,
        filter_size=3,
        num_filters=[12, 12],
        num_flat_units=[256, 128],
        drop_prob=0,
        optimiser='adam',
        learning_rate=1e-4,
        epochs=20,
        batch_size=32)

    train_a_model(
        "4b_hist4",
        do_augmentation=True,
        history_length=4,
        sequentialise=False,
        use_lstm=False,
        filter_size=3,
        num_filters=[12, 12],
        num_flat_units=[256, 128],
        drop_prob=0,
        optimiser='adam',
        learning_rate=1e-4,
        epochs=20,
        batch_size=32)

    train_a_model(
        "4c_hist4S",
        do_augmentation=True,
        history_length=4,
        sequentialise=True,
        use_lstm=False,
        filter_size=3,
        num_filters=[12, 12],
        num_flat_units=[256, 128],
        drop_prob=0,
        optimiser='adam',
        learning_rate=1e-4,
        epochs=20,
        batch_size=32)

    train_a_model(
        "4d_hist4L",
        do_augmentation=True,
        history_length=4,
        sequentialise=True,
        use_lstm=True,
        filter_size=3,
        num_filters=[12, 12],
        num_flat_units=[256, 128],
        drop_prob=0,
        optimiser='adam',
        learning_rate=1e-4,
        epochs=20,
        batch_size=32)

    train_a_model(
        "5_big",
        do_augmentation=True,
        history_length=4,
        sequentialise=False,
        use_lstm=False,
        filter_size=5,
        num_filters=[16, 32, 64],
        num_flat_units=[512, 128],
        drop_prob=0.2,
        optimiser='adam',
        learning_rate=1e-4,
        epochs=50,
        batch_size=32)
