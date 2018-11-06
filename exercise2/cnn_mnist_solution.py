#%% Imports

from __future__ import print_function

import gzip
import json
import os
import pickle

import numpy as np

import tensorflow as tf

try:
    from tensorflow import keras
    
except:
    #maybe it's an old version?
    import keras
            

# %% Functions

def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data', recs=None):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    valid_x, valid_y = valid_set
    train_x, train_y = train_set

    if recs is not None:
        train_x, train_y, valid_x, valid_y, test_x, test_y = train_x[:recs], train_y[:recs], valid_x[:recs], valid_y[:recs], test_x[:recs], test_y[:recs]


    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(
        valid_y), test_x, one_hot(test_y)

#%% Part A ###############################################
    
def read_from_json(name):
    fname = os.path.join(".", "results", "results_" + name + ".json")

    if os.path.exists(fname):
        with open(fname, "r") as fh:
            results = json.load(fh)
    
    return results

def train_and_validate(x_train, y_train, x_valid, y_valid, num_epochs, lr,
                       num_filters, kernel_width, batch_size, save_to=None):
    
    #TODO (STILL): change to tf
    
    #seems to be required to avoid weird behavior on retraining
    tf.reset_default_graph()
    
    model = keras.models.Sequential([
        keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(kernel_width, kernel_width),
            input_shape=(28, 28, 1),
            padding='same',
            activation='relu',
            data_format='channels_last'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(kernel_width, kernel_width),
            padding='same',
            activation='relu',
            data_format='channels_last'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=None),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    if save_to is not None:
        path = os.path.join(".", "results")
        os.makedirs(path, exist_ok=True)

        fname = os.path.join(path, "results_" + save_to + ".json")

#        if os.path.exists(fname):
#            with open(fname, "r") as fh:
#                results = json.load(fh)
#
#            return results["learning_curve"], model

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(x_valid, y_valid))

    learning_curve = [1 - a for a in history.history['val_acc']]

    if save_to is not None:
        results = dict()
        results["name"] = save_to
        results["lr"] = lr
        results["kernel_width"] = kernel_width
        results["num_filters"] = num_filters
        results["batch_size"] = batch_size
        results["learning_curve"] = learning_curve

        with open(fname, "w") as fh:
            json.dump(results, fh)

    return learning_curve, model


def test(x_test, y_test, model):

    return 1 - model.evaluate(x_test, y_test)[1]


if __name__ == '__main__':
    # %% Load Data

    # put this in ~/data so i'm not downloading the same thing into 100 different directories        
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(os.path.expanduser(os.sep.join(["~","data"])))
    
    
    # %% PART B ################################################
    
    # test learning rates with a 5x5 kernel
    
    for i in range(1, 5):
        lr = 10**-i
        train_and_validate(
            x_train,
            y_train,
            x_valid,
            y_valid,
            num_epochs=20,
            lr=lr,
            num_filters=16,
            kernel_width=5,
            batch_size=64,
            save_to='lr_10e-' + str(i)
        )
        
    # %% PART C ################################################
    
    # kernel sizes
    
    for i in (1, 3, 5, 7):
        train_and_validate(
            x_train,
            y_train,
            x_valid,
            y_valid,
            num_epochs=3,
            lr=0.01, #TODO
            num_filters=16,
            kernel_width=i,
            batch_size=64,
            save_to='kf_' + str(i)
        )
    
