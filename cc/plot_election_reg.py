import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

blue = "#377eb8"
purple = "#984ea3"
orange = "#ff7f00"
pink = "#f781bf"
pinks = ['#fccfe7', '#f99fcf', '#f66fb7', '#f33f9f', '#f12793']
purples = ['#984ea3','#a45aaf', '#ad6bb8', '#b77cc0','#c08dc8']
oranges = ['#ffb366','#ff9933', '#ff7f00', '#e67300', '#cc6600']
blues = ['#377eb8','#5697cd', '#7dafd8', '#a5c8e4']

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
font = {'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
font = {'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

epochs = 5000
seed = 0
reg_cc = 0.0
reg_dir = 0.0
lr_cc = 0.5
lr_dir = 0.1

for reg in [0.0, 1.0, 5.0, 10.0, 30.0]:
    reg_cc = reg
    reg_dir = reg
    results_dir = '../out/glm_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed)

    results = pd.read_pickle(results_dir)
    temp = results['CC_L2_err']

    orange = oranges.pop()
    if reg==0.0:
        plt.plot('train_step', 'CC_L2_err', data=results, color=blues[0])
    plt.plot('train_step', 'Dir_L2_err', data=results, color=orange)

    print("best CC L2", results["CC_L2_err"].min())
    temp = results["Dir_L2_err"].to_numpy()
    temp = [x.numpy() for x in temp]
    temp = np.array(temp)
    print("best Dir L2", temp[np.isfinite(temp)].min())
    temp = results["Dir_L1_err"].to_numpy()
    temp = [x.numpy() for x in temp]
    temp = np.array(temp)
    print("best Dir L1", temp[np.isfinite(temp)].min())


lr_cc = 0.01
lr_dir = 0.01
dp_rate = 0.0

for reg in [0.0, 1.0, 10.0, 100.0, 1000.0]:
    reg_cc = reg
    reg_dir = reg
    results_dir = '../out/mlp_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}_dp{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, dp_rate)

    results = pd.read_pickle(results_dir)
    temp = results['CC_L2_err']

    pink = pinks.pop()
    if reg==0.0:
        plt.plot('train_step', 'CC_L2_err', data=results, color=purples[0])
    plt.plot('train_step', 'Dir_L2_err', data=results, color=pink)

    print("best CC L2", results["CC_L2_err"].min())
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
pink = "#f781bf"

custom_lines = [Line2D([0], [0], lw=3, color=orange),
                Line2D([0], [0], lw=3, color=pink),
                Line2D([0], [0], lw=3, color=blue),
                Line2D([0], [0], lw=3, color=purple)]
plt.legend(custom_lines, ['Dir linear', 'Dir MLP', 'CC linear', 'CC MLP'])

plt.ylim(0.07,0.3)
plt.xlim(-200,5000)
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.tight_layout()
plt.savefig('../out/reg_election.pdf')
plt.show()

