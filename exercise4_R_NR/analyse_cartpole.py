# %%
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 16:57:35 2018

@author: AlexR
"""

import matplotlib.pyplot as plt
import json
import os
import matplotlib as mpl

HYPERS_FNM = 'hypers.json'
TEST_FNM = 'results.json'
MODELS_DIR = 'cartpole'

# %% lOAD UP ALL RESULTS

results = dict()


def get_model_results(directory):
    with open(os.path.join(MODELS_DIR, directory, HYPERS_FNM), "r") as fh:
        hypers = json.load(fh)
    with open(os.path.join(MODELS_DIR, directory, TEST_FNM), "r") as fh:
        test = json.load(fh)
    results[directory] = {'hypers': hypers, 'test': test}


get_model_results('1_basic')
get_model_results('2_epsdecay')
get_model_results('3_boltzmann')
get_model_results('4_doubleqb')

# %% Plot training curves

mpl.rcParams['savefig.dpi'] = 150


def doplot(data, saveto):

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig, ax2 = plt.subplots()

    ax2.set_title('Cartpole results')
    ax2.boxplot(
        [results[key]['test']['episode_rewards'] for key, label in data],
        labels=[label for key, label in data])
    ax2.set_ylabel('Episode rewards')

    fig.tight_layout()
    fig.savefig(saveto)
    # plt.show()



doplot([['1_basic', 'Default Model'], ['2_epsdecay', 'Epsilon Decay'], ['3_boltzmann', 'Boltzmann Exp\'n'], ['4_doubleqb', 'Double Q']],
       'figs/cartpole_test.png')