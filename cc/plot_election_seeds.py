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
pinks = ['#f781bf', '#f452a9', '#fab2d9', '#f12291']
purples = ['#984ea3','#a45aaf', '#ad6bb8', '#b77cc0','#c08dc8']
oranges = ['#ff7f00','#ff9933', '#ffb366', '#ffcc99']
blues = ['#377eb8','#5697cd', '#7dafd8', '#a5c8e4']

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
font = {'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

epochs = 1000
reg_cc = 0.0
reg_dir = 0.0
lr_cc = 0.5
lr_dir = 0.1
dp_rate = 0.0
notes = ''
for seed in [1, 2, 3, 4]:
    results_dir = '../out/glm_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, notes)
    results = pd.read_pickle(results_dir)

    blue = blues.pop()
    orange = oranges.pop()
    plt.plot(results['train_step'], results['CC_L2_err'], color=blue)
    plt.plot(results['train_step'], results['Dir_L2_err'], color=orange)

    temp = results["Dir_L2_err"].to_numpy()
    temp = [x.numpy() for x in temp]
    temp = np.array(temp)
    print("best Dir L2", temp[np.isfinite(temp)].min())
    temp = results["Dir_L1_err"].to_numpy()
    temp = [x.numpy() for x in temp]
    temp = np.array(temp)
    print("best Dir L1", temp[np.isfinite(temp)].min())


seed = 0
epochs = 500
reg_cc = 0.0
reg_dir = 0.0
lr_cc = 0.01
lr_dir = 0.01
for seed in [1, 2, 3, 4]:
    results_dir = '../out/mlp_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}_dp{}{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, dp_rate, notes)
    results = pd.read_pickle(results_dir)

    purple = purples.pop()
    pink = pinks.pop()
    plt.plot(results['train_step'], results['CC_L2_err'], color=purple)
    plt.plot(results['train_step'], results['Dir_L2_err'], color=pink)

    temp = results["Dir_L2_err"].to_numpy()
    temp = [x.numpy() for x in temp]
    temp = np.array(temp)
    print("best Dir L2", temp[np.isfinite(temp)].min())
    temp = results["Dir_L1_err"].to_numpy()
    temp = [x.numpy() for x in temp]
    temp = np.array(temp)
    print("best Dir L1", temp[np.isfinite(temp)].min())



blue = "#377eb8"
purple = "#984ea3"
orange = "#ff7f00"
pink = "#a65628"
pink = "#f781bf"

custom_lines = [Line2D([0], [0], lw=3, color=orange),
                Line2D([0], [0], lw=3, color=pink),
                Line2D([0], [0], lw=3, color=blue),
                Line2D([0], [0], lw=3, color=purple)]
plt.legend(custom_lines, ['Dir linear', 'Dir MLP', 'CC linear', 'CC MLP'])


plt.ylim(0.07,0.3)
plt.xlim(-10,400)
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.tight_layout()
plt.savefig('../out/seeds_election.pdf')
plt.show()

