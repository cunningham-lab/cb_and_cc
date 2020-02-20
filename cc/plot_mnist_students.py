import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
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
reg_cc = 0.0
reg_dir = 0.0
lr_dir = 0.01
lr_cc = 0.1
lr_xe = 0.1
t_adj = 1.0
soft_loss = 1.0
hard_loss = 0.0
BATCH_SIZE = 1000
notes = ''
omit3 = False
results_dir = '../out/mnist_student_temp{}_omit3{}_lrcc{}_lrdir{}_lrxe{}_soft{}_hard{}_batch{}{}.pkl'.format(
    t_adj, omit3, lr_cc, lr_dir, lr_xe, soft_loss, hard_loss, BATCH_SIZE, notes)
results = pd.read_pickle(results_dir)
print("best CC", results["CC_misclass"].min())
print("best XE", results["XE_misclass"].min())
print("best CC L2", results["CC_L2_err"].min())
print("best XE L2", results["XE_L2_err"].min())
temp = results["Dir_L2_err"].to_numpy()
temp = [x.numpy() for x in temp]
temp = np.array(temp)
print("best Dir L2", temp[np.isfinite(temp)].min())
print("best Dir", results["Dir_misclass"].min())
plt.plot('epoch', 'CC_L2_err', data=results, color=purple)
plt.plot('epoch', 'XE_L2_err', data=results, color=brown)
plt.plot('epoch', 'Dir_L2_err', data=results, color=pink)

soft_loss = 0.0
hard_loss = 1.0
t_adj = 1.0
results_dir = '../out/mnist_student_temp{}_omit3{}_lrcc{}_lrdir{}_lrxe{}_soft{}_hard{}_batch{}{}.pkl'.format(
    t_adj, omit3, lr_cc, lr_dir, lr_xe, soft_loss, hard_loss, BATCH_SIZE, notes)
results = pd.read_pickle(results_dir)
plt.plot('epoch', 'XE_L2_err', data=results, color='black')

soft_loss = 1.0
hard_loss = 0.0
t_adj=5.0
results_dir = '../out/mnist_student_temp{}_omit3{}_lrcc{}_lrdir{}_lrxe{}_soft{}_hard{}_batch{}{}.pkl'.format(
    t_adj, omit3, lr_cc, lr_dir, lr_xe, soft_loss, hard_loss, BATCH_SIZE, notes)
results = pd.read_pickle(results_dir)
print("best CC", results["CC_misclass"].min())
print("best XE", results["XE_misclass"].min())
print("best CC L2", results["CC_L2_err"].min())
print("best XE L2", results["XE_L2_err"].min())
temp = results["Dir_L2_err"].to_numpy()
temp = [x.numpy() for x in temp]
temp = np.array(temp)
print("best Dir L2", temp[np.isfinite(temp)].min())
print("best Dir", results["Dir_misclass"].min())
plt.plot('epoch', 'CC_L2_err', data=results, color=purple, linestyle='dashed')
plt.plot('epoch', 'XE_L2_err', data=results, color=brown, linestyle='dashed')
plt.plot('epoch', 'Dir_L2_err', data=results, color=pink, linestyle='dashed')

t_adj=10.0
results_dir = '../out/mnist_student_temp{}_omit3{}_lrcc{}_lrdir{}_lrxe{}_soft{}_hard{}_batch{}{}.pkl'.format(
    t_adj, omit3, lr_cc, lr_dir, lr_xe, soft_loss, hard_loss, BATCH_SIZE, notes)
results = pd.read_pickle(results_dir)
print("best CC", results["CC_misclass"].min())
print("best XE", results["XE_misclass"].min())
print("best CC L2", results["CC_L2_err"].min())
print("best XE L2", results["XE_L2_err"].min())
temp = results["Dir_L2_err"].to_numpy()
temp = [x.numpy() for x in temp]
temp = np.array(temp)
print("best Dir L2", temp[np.isfinite(temp)].min())
print("best Dir", results["Dir_misclass"].min())
plt.plot('epoch', 'CC_L2_err', data=results, color=purple, linestyle='dotted')
plt.plot('epoch', 'XE_L2_err', data=results, color=brown, linestyle='dotted')
plt.plot('epoch', 'Dir_L2_err', data=results, color=pink, linestyle='dotted')

custom_lines = [Line2D([0], [0], lw=3, color=pink),
                Line2D([0], [0], lw=3, color=brown),
                Line2D([0], [0], lw=3, color=purple),
                Line2D([0], [0], lw=3, color='black'),
                Line2D([0], [0], lw=3, color='gray'),
                Line2D([0], [0], lw=3, color='gray', linestyle='dashed'),
                Line2D([0], [0], lw=3, color='gray', linestyle='dotted')]
plt.legend(custom_lines, ['Dir', 'soft XE', 'CC', 'hard XE', 'T=1', 'T=5', 'T=10'], loc='upper right')

plt.xlim(-10,400)
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.tight_layout()
plt.savefig('../out/students_test_err_temps.pdf')
plt.show()

