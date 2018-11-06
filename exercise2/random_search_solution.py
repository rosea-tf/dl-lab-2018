import argparse
from hpbandster.core.worker import Worker
import ConfigSpace as CS
from hpbandster.optimizers import RandomSearch
import hpbandster.core.nameserver as hpns
import logging
import os
import pickle
import numpy as np

from cnn_mnist_solution import mnist, train_and_validate

logging.basicConfig(level=logging.WARNING)


# %% Function definitions

class MyWorker(Worker):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = mnist(
            os.path.expanduser(os.sep.join(["~", "data"])), recs=500)

        pass

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
        lr = config["lr"]
        num_filters = config["num_filters"]
        batch_size = config["batch_size"]
        filter_size = config["filter_size"]
        epochs = budget

        learning_curve, model = train_and_validate(
            self.x_train, self.y_train, self.x_valid, self.y_valid, epochs, lr, num_filters, filter_size, batch_size)

        return ({
            # this is the a mandatory field to run hyperband
            'loss': learning_curve[-1],
            'info': {}  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()

        # not sure what default_value does...
        lr = CS.hyperparameters.UniformFloatHyperparameter(
            'lr', lower=1e-4, upper=1e-1, default_value=1e-2, log=True)
        batch_size = CS.hyperparameters.UniformIntegerHyperparameter(
            'batch_size', lower=16, upper=128, default_value=64, log=True)
        num_filters = CS.hyperparameters.UniformIntegerHyperparameter(
            'num_filters', lower=2**3, upper=2**6, default_value=2**4, log=True)
        filter_size = CS.hyperparameters.CategoricalHyperparameter('filter_size', [
                                                                   3, 5])

        config_space.add_hyperparameters(
            [lr, batch_size, num_filters, filter_size])

        return config_space

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

rs = RandomSearch(configspace=w.get_configspace(),
                  run_id='example1', nameserver='127.0.0.1',
                  min_budget=1, max_budget=1)

# res = rs.run(n_iterations=args.n_iterations)
res = rs.run(n_iterations=10)

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
pickle.dump(res, open('results/hypersearch.pkl', 'wb'))


# %% Train the Incumbent (on combined T+V)

incumbent_config = id2config[incumbent]['config']

lr = incumbent_config["lr"]
num_filters = incumbent_config["num_filters"]
batch_size = incumbent_config["batch_size"]
filter_size = incumbent_config["filter_size"]
epochs = 50 # should be enough

# we use the test set as 'validation' here purely to retrieve the final performance
learning_curve, incumbent_model = train_and_validate(np.vstack([w.x_train, w.x_valid]), np.vstack([w.y_train, w.y_valid]), 
w.x_test, w.y_test, epochs, lr, num_filters, filter_size, batch_size, save_to='incumbent')