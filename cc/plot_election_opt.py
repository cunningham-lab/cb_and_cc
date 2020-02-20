import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

blue = "#377eb8"
purple = "#984ea3"
orange = "#ff7f00"
pink = "#a65628"
pink = "#f781bf"

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
font = {'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

seed = 0
epochs = 500
reg_cc = 0.0
reg_dir = 0.0
lr_cc = 0.5
lr_dir = 0.1
notes = ''
results_dir = '../out/glm_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, notes)
results = pd.read_pickle(results_dir)
plt.plot('train_step', 'CC_loglik', data=results, color=blue)
plt.plot('train_step', 'Dir_loglik', data=results, color=orange)

lr_cc = 0.1
lr_dir = 0.1
notes = 'RMSprop'
results_dir = '../out/glm_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, notes)
results = pd.read_pickle(results_dir)
plt.plot('train_step', 'CC_loglik', data=results, color=blue, linestyle='dashed')
plt.plot('train_step', 'Dir_loglik', data=results, color=orange, linestyle='dashed')

lr_cc = 10.0
lr_dir = 10.0
notes = 'Adadelta'
results_dir = '../out/glm_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, notes)
results = pd.read_pickle(results_dir)
plt.plot('train_step', 'CC_loglik', data=results, color=blue, linestyle='dotted')

lr_cc = 50.0
lr_dir = 50.0
notes = 'Adadelta'
results_dir = '../out/glm_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, notes)
results = pd.read_pickle(results_dir)
results['Dir_loglik'][210:1000] = 1e20 # Afterwards we get +- inf messing up the plot
plt.plot('train_step', 'Dir_loglik', data=results, color=orange, linestyle='dotted')

seed = 0
epochs = 1000
reg_cc = 0.0
reg_dir = 0.0
lr_cc = 0.01
lr_dir = 0.01
dp_rate = 0.0
notes = ''
results_dir = '../out/mlp_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}_dp{}{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, dp_rate, notes)
results = pd.read_pickle(results_dir)
plt.plot('train_step', 'loglik_cc', data=results, color=purple)
plt.plot('train_step', 'loglik_dir', data=results, color=pink)

lr_cc = 0.05
lr_dir = 0.05
dp_rate = 0.0
notes = 'RMSprop'
results_dir = '../out/mlp_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}_dp{}{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, dp_rate, notes)
results = pd.read_pickle(results_dir)
plt.plot('train_step', 'loglik_cc', data=results, color=purple, linestyle='dashed')

lr_cc = 0.01
lr_dir = 0.01
dp_rate = 0.0
notes = 'RMSprop'
results_dir = '../out/mlp_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}_dp{}{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, dp_rate, notes)
results = pd.read_pickle(results_dir)
plt.plot('train_step', 'loglik_dir', data=results, color=pink, linestyle='dashed')

lr_cc = 1.0
lr_dir = 1.0
dp_rate = 0.0
notes = 'Adadelta'
results_dir = '../out/mlp_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}_dp{}{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, dp_rate, notes)
results = pd.read_pickle(results_dir)
plt.plot('train_step', 'loglik_cc', data=results, color=purple, linestyle='dotted')

lr_cc = 5.0
lr_dir = 5.0
dp_rate = 0.0
notes = 'Adadelta'
results_dir = '../out/mlp_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}_dp{}{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, dp_rate, notes)
results = pd.read_pickle(results_dir)
results['loglik_dir'][70:-1] = 1e20
plt.plot('train_step', 'loglik_dir', data=results, color=pink, linestyle='dotted')


custom_lines = [Line2D([0], [0], lw=3, color=orange),
                Line2D([0], [0], lw=3, color=pink),
                Line2D([0], [0], lw=3, color=blue),
                Line2D([0], [0], lw=3, color=purple),
                Line2D([0], [0], lw=3, color='gray'),
                Line2D([0], [0], lw=3, color='gray', linestyle='dashed'),
                Line2D([0], [0], lw=3, color='gray', linestyle='dotted')]
plt.legend(custom_lines, ['Dir linear', 'Dir MLP', 'CC linear', 'CC MLP', 'Adam', 'RMSprop', 'Adadelta'])

plt.xlabel('epoch')
plt.ylabel('LogLik')
plt.xlim(-10,400)
plt.ylim(-4e3,8e3)
plt.tight_layout()
plt.savefig('../out/optimizers.pdf')
plt.show()

