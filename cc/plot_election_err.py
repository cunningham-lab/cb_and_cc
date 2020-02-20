import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

blue = "#377eb8"
purple = "#984ea3"
orange = "#ff7f00"
brown = "#a65628"
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

print("best CC L2", results["CC_L2_err"].min())
print("best CC L1", results["CC_L1_err"].min())
temp = results["Dir_L2_err"].to_numpy()
temp = [x.numpy() for x in temp]
temp = np.array(temp)
print("best Dir L2", temp[np.isfinite(temp)].min())
temp = results["Dir_L1_err"].to_numpy()
temp = [x.numpy() for x in temp]
temp = np.array(temp)
print("best Dir L1", temp[np.isfinite(temp)].min())


plt.plot('train_step', 'CC_L2_err', data=results, color=blue)
plt.plot('train_step', 'CC_L1_err', data=results, color=blue, linestyle='dotted')
plt.plot('train_step', 'Dir_L2_err', data=results, color=orange)
plt.plot('train_step', 'Dir_L1_err', data=results, color=orange, linestyle='dotted')




seed = 0
epochs = 1000
reg_cc = 0.0
reg_dir = 0.0
lr_cc = 0.01
lr_dir = 0.01
dp_rate = 0.0

results_dir = '../out/mlp_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}_dp{}{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, dp_rate, notes)


results = pd.read_pickle(results_dir)

print("best CC L2", results["CC_L2_err"].min())
print("best CC L1", results["CC_L1_err"].min())
temp = results["Dir_L2_err"].to_numpy()
temp = [x.numpy() for x in temp]
temp = np.array(temp)
print("best Dir L2", temp[np.isfinite(temp)].min())
temp = results["Dir_L1_err"].to_numpy()
temp = [x.numpy() for x in temp]
temp = np.array(temp)
print("best Dir L1", temp[np.isfinite(temp)].min())

plt.plot('train_step', 'CC_L2_err', data=results, color=purple)
plt.plot('train_step', 'Dir_L2_err', data=results, color=pink)
plt.plot('train_step', 'CC_L1_err', data=results, color=purple, linestyle='dotted')
plt.plot('train_step', 'Dir_L1_err', data=results, color=pink, linestyle='dotted')


custom_lines = [Line2D([0], [0], lw=3, color=orange),
                Line2D([0], [0], lw=3, color=pink),
                Line2D([0], [0], lw=3, color=blue),
                Line2D([0], [0], lw=3, color=purple),
                Line2D([0], [0], lw=3, color='gray'),
                Line2D([0], [0], lw=3, color='gray', linestyle='dotted')]
plt.legend(custom_lines, ['Dir linear', 'Dir MLP', 'CC linear', 'CC MLP', 'RMSE', 'MAE'])

plt.xlim(-10,400)
plt.ylim(0.04,0.3)
plt.xlabel('epoch')
plt.ylabel('test error')
plt.savefig('../out/test_error_glm_mlp.pdf')
plt.show()
