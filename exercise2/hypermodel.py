# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:37:56 2018

@author: AlexR
"""
import pickle
import numpy as np
from pyswarm import pso
from cnn_mnist_solution_tf import mnist, create_the_model, results_dump
import os


# %% Hyperparameter Model ###################################

np.random.seed(123)

# retrieve results of hyperparameter research

res = pickle.load(open('results/hypersearch.pkl','rb'))

# form into dataset, X, Y
hyper_dataset = [[run.config_id, run.loss, res.get_id2config_mapping()[run.config_id]['config']] for run in res.get_all_runs()]
hyper_X = np.array([list(r[2].values()) for r in hyper_dataset])
hyper_y = np.array([r[1] for r in hyper_dataset]).reshape(-1,1)

# train model of loss <- hypers
from keras.models import Sequential
from keras.layers import Dense, Activation

hyper_model = Sequential([
    Dense(20, input_shape=(4,)),
    Activation('tanh'),
    Dense(10),
    Activation('tanh'),
    Dense(1)
])

hyper_model.compile(optimizer='rmsprop',
              loss='mse')

hyper_model.fit(hyper_X, hyper_y, epochs=50)

#%% PSO to find argmin x ###################################


def pso_obj_fn(x):
    return np.sum(hyper_model.predict(np.array(x).reshape(1,-1)))


pso_results = pso(pso_obj_fn,
            swarmsize=50,
            lb=[1e-4, 16, 3, 2 ** 3],
            ub=[1e-1, 128, 5, 2 ** 6],
            omega=0.7,
            phip=1.49618,
            phig=1.49618,
            maxiter=1000)

print("PSO done!", pso_results)

#%% Train MNIST model ###################################

x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(
            os.path.expanduser(os.sep.join(["~", "data"])))

lr = pso_results[0][0]
batch_size, filter_size, num_filters = [int(np.round(h, 0)) for h in pso_results[0][1:]]

epochs = 50  # should be enough

print("Training experimental model!!!")

experimental_model = create_the_model(x_train, y_train, filter_size,
                                   num_filters)

# we use the test set as 'validation' here purely to retrieve the final performance
experimental_model.train(
    np.vstack([x_train, x_valid]), np.vstack([y_train, y_valid]),
    x_test, y_test, lr, epochs, batch_size)
experimental_model.save('experimental')

results_path = os.path.join(".", "results")

results_dump(results_path, "experimental", lr, filter_size, num_filters,
             batch_size, experimental_model.valid_accs)