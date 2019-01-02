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
import pandas as pd

HYPERS_FNM = 'hypers.json'
TEST_FNM = 'results.json'
TEST_SM_FNM = 'results_sm.json'
MODELS_DIR = 'racecar'

# %% lOAD UP ALL RESULTS

results = dict()


def get_model_results(directory, caption):
    with open(os.path.join(MODELS_DIR, directory, HYPERS_FNM), "r") as fh:
        hypers = json.load(fh)
    with open(os.path.join(MODELS_DIR, directory, TEST_FNM), "r") as fh:
        test = json.load(fh)
    with open(os.path.join(MODELS_DIR, directory, TEST_SM_FNM), "r") as fh:
        test_sm = json.load(fh)

    results[directory] = {'hypers': hypers, 'test': test, 'test_sm': test_sm, 'caption': caption}


get_model_results('1_basic', 'Default Model')
get_model_results('2_epsdecay', 'Epsilon Decay')
get_model_results('3_boltzmann', 'Boltzmann Exp\'n')
get_model_results('4_doubleq', 'Double Q')
get_model_results('7_history', 'History Frame')
get_model_results('8_difframe', 'Difference Frame')
get_model_results('9_diffpenalty', 'DF + Penalty')
get_model_results('10_dpbig', 'DF+P, Large Net')
get_model_results('12_dpnoskip', 'DF+P, No Skip')

# %% Plot training curves

mpl.rcParams['savefig.dpi'] = 150


def doplot(data, saveto):

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig, ax2 = plt.subplots()
    ax2.boxplot(
        [results[key]['test']['episode_rewards'] for key in data] +
        [results[key]['test_sm']['episode_rewards'] for key in data],
        labels=[results[key]['caption'] for key in data
                ] + [results[key]['caption'] + '\n(softmax actions)' for key in data])
    ax2.set_ylim((250, 950))
    ax2.set_ylabel('Episode rewards')

    fig.tight_layout()
    fig.savefig(saveto)
    # plt.show()

#%%

doplot(['1_basic', '4_doubleq'],
       'report/figs/racecar_1.png')

doplot(['2_epsdecay', '3_boltzmann'],
       'report/figs/racecar_2.png')

doplot(['7_history', '8_difframe'],
       'report/figs/racecar_3.png')

doplot(['9_diffpenalty', '10_dpbig'],
       'report/figs/racecar_4.png')



#%% Table of results

captions = [model['caption'] for model in results.values()]
test_means = [round(model['test']['mean'], 1) for model in results.values()]
test_sm_means = [round(model['test_sm']['mean'], 1) for model in results.values()]

df = pd.DataFrame({'Captions': captions, 'Avg Test Score': test_means, 'Avg Score (Softmax)': test_sm_means})

with open('report/figs/result_table.tex','w') as fh:
    fh.write(df.to_latex(index=False))