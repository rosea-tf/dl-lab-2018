# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 16:57:35 2018

@author: AlexR
"""

import matplotlib.pyplot as plt
import json
import os
import matplotlib as mpl

HYPERS_FNM = 'cnn_hypers.json'
TRAIN_FNM = 'cnn_results.json'
TEST_FNM = 'test_results.json'
MODELS_DIR = 'models'

# %% lOAD UP ALL RESULTS

results = dict()

def get_model_results(directory):
    with open(os.path.join(MODELS_DIR, directory, HYPERS_FNM), "r") as fh:
        hypers = json.load(fh)
    with open(os.path.join(MODELS_DIR, directory, TRAIN_FNM), "r") as fh:
        train = json.load(fh)
    with open(os.path.join(MODELS_DIR, directory, TEST_FNM), "r") as fh:
        test = json.load(fh)
    
    results[directory] = {'hypers':hypers, 'train': train, 'test': test}


get_model_results('1_basic')
get_model_results('2_augmented')
get_model_results('3a_dropout05')
get_model_results('3b_dropout10')
get_model_results('3c_dropout20')
get_model_results('4a_hist2')
get_model_results('4b_hist4')
get_model_results('4c_hist4S')
get_model_results('4d_hist4L')
get_model_results('incumbent')

# %% Plot training curves

mpl.rcParams['savefig.dpi'] = 150

def accs_plot(dictionary, ax, label):
    line, = ax.plot(dictionary['train_accs'], label=label)
    ax.plot(dictionary['valid_accs'], line.get_color(), linestyle=':')


def doplot(data, saveto):
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    for key, label in data:
        accs_plot(results[key]['train'], ax1, label)
    
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    ax2.boxplot([results[key]['test']['episode_rewards'] for key, label in data],
        labels=[label for key, label in data])
    ax2.set_ylabel('Episode rewards')
    
    fig.tight_layout()
    fig.savefig(saveto)
    plt.show()


doplot([['1_basic','Initial Data'],['2_augmented','Balanced Data']], 'figs/balance.png')

doplot([['2_augmented','Dropout 0.00'],
['3a_dropout05','Dropout 0.05'],
['3b_dropout10','Dropout 0.10'],
['3c_dropout20','Dropout 0.20']], 'figs/dropout.png')

doplot([['2_augmented','History = 1'],
['4a_hist2','History = 2'],
['4b_hist4','History = 4'],
['4c_hist4S','Sequential']], 'figs/history.png')
    
doplot([['2_augmented','Basic Model'],['incumbent','Best BOHB Model']], 'figs/bohb.png')