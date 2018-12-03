import argparse
from hpbandster.core.worker import Worker
import ConfigSpace as CS
from hpbandster.optimizers import RandomSearch
from hpbandster.optimizers import BOHB
import hpbandster.core.nameserver as hpns
import logging
import os
import pickle
import numpy as np
import gzip

from train_agent import create_a_model, train_a_model

logging.basicConfig(level=logging.WARNING)

# %% Function definitions


class MyWorker(Worker):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        with gzip.open(os.path.join('datasets', 'xy3.pkl.gz'), 'rb') as fh:
            self.X_train, self.y_train, self.X_valid, self.y_valid = pickle.load(
                fh)

    def compute(self, config, budget, **kwargs):
        """
        Evaluates the configuration on the defined budget and returns the validation performance.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        history_length = config['history_length']
        sequentialise = False if history_length <= 1 else config[
            'sequentialise']
        use_lstm = False if not sequentialise else config['lstm']
        filter_size = config["filter_size"]
        num_filters = [
            config["num_filters"] for i in range(config["num_convlayers"])
        ]
        num_flat_units = [
            config['num_flat_units'] for i in range(config['num_flatlayers'])
        ]
        drop_prob = config['drop_prob']
        learning_rate = config["lr"]

        epochs = int(budget)
        batch_size = config["batch_size"]

        X_train = self.X_train[..., :history_length]
        X_valid = self.X_valid[..., :history_length]
        y_train = self.y_train
        y_valid = self.y_valid

        model = create_a_model(
            X_shape=X_train.shape,
            y_shape=y_train.shape,
            sequentialise=sequentialise,
            use_lstm=use_lstm,
            filter_size=filter_size,
            num_filters=num_filters,
            num_flat_units=num_flat_units,
            drop_prob=drop_prob,
            optimiser='adam',
            learning_rate=learning_rate)

        model.train(
            x_train=X_train,
            y_train=y_train,
            x_valid=X_valid,
            y_valid=y_valid,
            epochs=epochs,
            batch_size=batch_size)

        final_val_acc = 1 - model.valid_accs[-1]

        return ({
            # this is the a mandatory field to run hyperband
            'loss': final_val_acc,
            'info':
            {}  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()

        # not sure what default_value does...
        lr = CS.hyperparameters.UniformFloatHyperparameter(
            'lr', lower=1e-4, upper=1e-1, default_value=1e-2, log=True)
        batch_size = CS.hyperparameters.UniformIntegerHyperparameter(
            'batch_size', lower=16, upper=128, default_value=64, log=True)
        num_convlayers = CS.hyperparameters.UniformIntegerHyperparameter(
            'num_convlayers', lower=1, upper=4, default_value=2, log=True)
        num_flatlayers = CS.hyperparameters.UniformIntegerHyperparameter(
            'num_flatlayers', lower=1, upper=3, default_value=2, log=False)
        num_filters = CS.hyperparameters.UniformIntegerHyperparameter(
            'num_filters',
            lower=2**3,
            upper=2**5,
            default_value=2**4,
            log=True)
        num_flat_units = CS.hyperparameters.UniformIntegerHyperparameter(
            'num_flat_units',
            lower=2**6,
            upper=2**9,
            default_value=2**7,
            log=True)
        history_length = CS.hyperparameters.UniformIntegerHyperparameter(
            'history_length', lower=1, upper=4, default_value=2, log=True)

        drop_prob = CS.hyperparameters.UniformFloatHyperparameter(
            'drop_prob', lower=0, upper=0.4, default_value=0.1, log=False)

        filter_size = CS.hyperparameters.CategoricalHyperparameter(
            'filter_size', [3, 5])
        sequentialise = CS.hyperparameters.CategoricalHyperparameter(
            'sequentialise', [False, True])
        lstm = CS.hyperparameters.CategoricalHyperparameter(
            'lstm', [False, True])

        use_sequential = CS.InCondition(
            child=sequentialise, parent=history_length, values=[2, 3, 4])
        use_lstm = CS.InCondition(
            child=lstm, parent=sequentialise, values=[True])

        cs.add_hyperparameters([
            lr, batch_size, num_convlayers, num_flatlayers, num_filters,
            num_flat_units, history_length, drop_prob, filter_size,
            sequentialise, lstm
        ])
        cs.add_conditions([use_sequential, use_lstm])

        return cs


# %% Run hyper search

# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any
# HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = MyWorker(nameserver='127.0.0.1', run_id='example1')
w.run(background=True)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run RandomSearch, but that is not essential.
# The run method will return the `Result` that contains all runs performed.

# rs = RandomSearch(configspace=w.get_configspace(),
#                  run_id='example1', nameserver='127.0.0.1',
#                  min_budget=int(args.budget), max_budget=int(args.budget))

# rs = RandomSearch(
# configspace=w.get_configspace(),
# run_id='example1',
# nameserver='127.0.0.1',
# min_budget=6,
# max_budget=6)

rs = BOHB(
    configspace=w.get_configspace(),
    run_id='example1',
    nameserver='127.0.0.1',
    min_budget=3,
    max_budget=9)

# res = rs.run(n_iterations=args.n_iterations)
res = rs.run(n_iterations=50)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
rs.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds information about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the
# performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])

# for plotting in notebook
results_path = os.path.join(".", "results")
os.makedirs(results_path, exist_ok=True)
pickle.dump(res, open(os.path.join(results_path, "hypersearch.pkl"), 'wb'))

# %% Train the Incumbent (on combined T+V)

config = id2config[incumbent]['config']

history_length = config['history_length']
sequentialise = False if history_length <= 1 else config['sequentialise']
use_lstm = False if not sequentialise else config['lstm']
filter_size = config["filter_size"]
num_filters = [config["num_filters"] for i in range(config["num_convlayers"])]
num_flat_units = [
    config['num_flat_units'] for i in range(config['num_flatlayers'])
]
drop_prob = config['drop_prob']
learning_rate = config["lr"]

epochs = 30
batch_size = config["batch_size"]

print("Training final model!!!")

train_a_model(
    "incumbent",
    do_augmentation=True,
    history_length=history_length,
    sequentialise=sequentialise,
    use_lstm=use_lstm,
    filter_size=filter_size,
    num_filters=num_filters,
    num_flat_units=num_flat_units,
    drop_prob=drop_prob,
    optimiser='adam',
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size)
