
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# Set up the colors
blue = "#377eb8"
purple = "#984ea3"
orange = "#ff7f00"
brown = "#a65628"
pink = "#f781bf"

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

# Set up the window of the moving average:
W = 10
alpha = 0.3

# First we setup the functions to extract the info we need from the tensorboard files:
def extract_summaries(path, summary):
    steps = []
    evalscores = []
    SI = summary_iterator(path)
    for event in SI:
        for value in event.summary.value:
            if value.tag == summary:
                steps.append(event.step)
                evalscores.append(float(tensor_util.MakeNdarray(value.tensor)))

    res = pd.DataFrame({'Step': steps, 'EvalScore': evalscores})
    return res

summary_dqn = 'Evaluation score'

xlim = [0, 100.1]
plt.figure(figsize=(16,5))

plt.subplot(141)
env_name = 'Breakout'
summary_amn = env_name + 'Deterministic-v4' + '-EvaluationScore'
path_dqn, path_amn_xe, path_amn_cc = [
    './tensorboard/BreakoutDeterministic-v4/2020-06-18-233535/events.out.tfevents.1592537738.t136.267902.5.v2',
    './amn/tensorboard/BreakoutDeterministic-v4.loss=XE.T=1.02020-07-04-005632/events.out.tfevents.1593838597.t061.321542.1372.v2',
    './amn/tensorboard/BreakoutDeterministic-v4.loss=CC.T=1.02020-07-02-184433/events.out.tfevents.1593729885.t136.12550.1372.v2'
]

results = extract_summaries(path_dqn, summary_dqn)
results['Step'] = results['Step'] / 1e5 - 1
results['EvalScore_SMA'] = results['EvalScore'].rolling(window=W).mean()
plt.plot('Step', 'EvalScore', data=results, color='grey', alpha=alpha)
plt.plot('Step', 'EvalScore_SMA', data=results, color='grey')
print(env_name)
print(results['EvalScore'][-20:].mean())
print(results['EvalScore'][-20:].std())


results = extract_summaries(path_amn_cc, summary_amn)
results['Step'] = results['Step'] / 1e5 - 1
results['EvalScore_SMA'] = results['EvalScore'].rolling(window=W).mean()
plt.plot('Step', 'EvalScore', data=results, color=purple, alpha=alpha)
plt.plot('Step', 'EvalScore_SMA', data=results, color=purple)
print(env_name)
print(results['EvalScore'][-20:].mean())
print(results['EvalScore'][-20:].std())

results = extract_summaries(path_amn_xe, summary_amn)
results['Step'] = results['Step'] / 1e5 - 1
results['EvalScore_SMA'] = results['EvalScore'].rolling(window=W).mean()
plt.plot('Step', 'EvalScore', data=results, color=brown, alpha=alpha)
plt.plot('Step', 'EvalScore_SMA', data=results, color=brown)
print(env_name)
print(results['EvalScore'][-20:].mean())
print(results['EvalScore'][-20:].std())

custom_lines = [
                Line2D([0], [0], lw=3, color='grey'),
                Line2D([0], [0], lw=3, color=brown),
                Line2D([0], [0], lw=3, color=purple),
               ]
plt.legend(custom_lines, ['DQN', 'AMN', 'CC-AMN'])

plt.xticks([0, 50, 100])
plt.xlim(xlim)
plt.xlabel('Training Epoch')
plt.ylabel('Evaluation Score')
plt.title(env_name)
plt.yticks([0,300,600], rotation='vertical')
plt.ylim(0,600)


plt.subplot(142)
env_name = 'Atlantis'
summary_amn = env_name + 'Deterministic-v4' + '-EvaluationScore'
path_dqn, path_amn_xe, path_amn_cc = [
    './tensorboard/AtlantisDeterministic-v4/2020-06-23-233328/events.out.tfevents.1592969611.node079.25565.5.v2',
    './amn/tensorboard/AtlantisDeterministic-v4.loss=XE.T=1.02020-07-06-224509/events.out.tfevents.1594089925.t136.222222.1372.v2',
    './amn/tensorboard/AtlantisDeterministic-v4.loss=CC.T=1.02020-07-03-030925/events.out.tfevents.1593760170.t065.276564.1372.v2'
]

results = extract_summaries(path_dqn, summary_dqn)
results['Step']= results['Step'] / 1e5 - 1
results['EvalScore_SMA'] = results['EvalScore'].rolling(window=W).mean()
plt.plot('Step', 'EvalScore', data=results, color='grey', alpha=alpha)
plt.plot('Step', 'EvalScore_SMA', data=results, color='grey')
print(env_name)
print(results['EvalScore'][-20:].mean())
print(results['EvalScore'][-20:].std())


results = extract_summaries(path_amn_cc, summary_amn)
results['Step'] = results['Step'] / 1e5 - 1
results['EvalScore_SMA'] = results['EvalScore'].rolling(window=W).mean()
plt.plot('Step', 'EvalScore', data=results, color=purple, alpha=alpha)
plt.plot('Step', 'EvalScore_SMA', data=results, color=purple)
print(env_name)
print(results['EvalScore'][-20:].mean())
print(results['EvalScore'][-20:].std())

results = extract_summaries(path_amn_xe, summary_amn)
results['Step'] = results['Step'] / 1e5 - 1
results['EvalScore_SMA'] = results['EvalScore'].rolling(window=W).mean()
plt.plot('Step', 'EvalScore', data=results, color=brown, alpha=alpha)
plt.plot('Step', 'EvalScore_SMA', data=results, color=brown)
print(env_name)
print(results['EvalScore'][-20:].mean())
print(results['EvalScore'][-20:].std())

plt.xticks([0, 50, 100])
plt.xlim(xlim)
plt.xlabel('Training Epoch')
plt.title(env_name)
plt.yticks([0,25000,50000], rotation='vertical')
plt.ylim(0,50000)




plt.subplot(143)
env_name = 'Pong'
summary_amn = env_name + 'Deterministic-v4' + '-EvaluationScore'
path_dqn, path_amn_xe, path_amn_cc = [
    './tensorboard/PongDeterministic-v4/2020-06-23-234131/events.out.tfevents.1592970094.node076.26174.5.v2',
    './amn/tensorboard/PongDeterministic-v4.loss=XE.T=1.02020-07-05-030717/events.out.tfevents.1593932842.t060.256916.1372.v2',
    './amn/tensorboard/PongDeterministic-v4.loss=CC.T=1.02020-08-27-005949/events.out.tfevents.1598504401.t059.13585.1372.v2'
]

results = extract_summaries(path_dqn, summary_dqn)
results['Step']= results['Step'] / 1e5 - 1
results['EvalScore_SMA'] = results['EvalScore'].rolling(window=W).mean()
plt.plot('Step', 'EvalScore', data=results, color='grey', alpha=alpha)
plt.plot('Step', 'EvalScore_SMA', data=results, color='grey')
print(env_name)
print(results['EvalScore'][-20:].mean())
print(results['EvalScore'][-20:].std())


results = extract_summaries(path_amn_cc, summary_amn)
results['Step']= results['Step'] / 1e5 - 1
results['EvalScore_SMA'] = results['EvalScore'].rolling(window=W).mean()
plt.plot('Step', 'EvalScore', data=results, color=purple, alpha=alpha)
plt.plot('Step', 'EvalScore_SMA', data=results, color=purple)
print(env_name)
print(results['EvalScore'][-20:].mean())
print(results['EvalScore'][-20:].std())

results = extract_summaries(path_amn_xe, summary_amn)
results['Step'] = results['Step'] / 1e5 - 1
results['EvalScore_SMA'] = results['EvalScore'].rolling(window=W).mean()
plt.plot('Step', 'EvalScore', data=results, color=brown, alpha=alpha)
plt.plot('Step', 'EvalScore_SMA', data=results, color=brown)
print(env_name)
print(results['EvalScore'][-20:].mean())
print(results['EvalScore'][-20:].std())

plt.xticks([0, 50, 100])
plt.xlim(xlim)
plt.xlabel('Training Epoch')
plt.title(env_name)
plt.yticks([-21,0,21], rotation='vertical')
plt.ylim(-21,22)




plt.subplot(144)
env_name = 'SpaceInvaders'
summary_amn = env_name + 'Deterministic-v4' + '-EvaluationScore'
path_dqn, path_amn_xe, path_amn_cc = [
    './tensorboard/SpaceInvadersDeterministic-v4/2020-06-23-233328/events.out.tfevents.1592969610.node078.27845.5.v2',
    './amn/tensorboard/SpaceInvadersDeterministic-v4.loss=XE.T=1.02020-07-07-071001/events.out.tfevents.1594120209.t065.32036.1372.v2',
    './amn/tensorboard/SpaceInvadersDeterministic-v4.loss=CC.T=1.02020-08-27-005949/events.out.tfevents.1598504401.t061.12339.1372.v2'
]

results = extract_summaries(path_dqn, summary_dqn)
results['Step'] = results['Step'] / 1e5 - 1
results['EvalScore_SMA'] = results['EvalScore'].rolling(window=W).mean()
plt.plot('Step', 'EvalScore', data=results, color='grey', alpha=alpha)
plt.plot('Step', 'EvalScore_SMA', data=results, color='grey')
print(env_name)
print(results['EvalScore'][-20:].mean())
print(results['EvalScore'][-20:].std())


results = extract_summaries(path_amn_cc, summary_amn)
results['Step'] = results['Step'] / 1e5 - 1
results['EvalScore_SMA'] = results['EvalScore'].rolling(window=W).mean()
plt.plot('Step', 'EvalScore', data=results, color=purple, alpha=alpha)
plt.plot('Step', 'EvalScore_SMA', data=results, color=purple)
print(env_name)
print(results['EvalScore'][-20:].mean())
print(results['EvalScore'][-20:].std())

results = extract_summaries(path_amn_xe, summary_amn)
results['Step'] = results['Step'] / 1e5 - 1
results['EvalScore_SMA'] = results['EvalScore'].rolling(window=W).mean()
plt.plot('Step', 'EvalScore', data=results, color=brown, alpha=alpha)
plt.plot('Step', 'EvalScore_SMA', data=results, color=brown)
print(env_name)
print(results['EvalScore'][-20:].mean())
print(results['EvalScore'][-20:].std())

plt.xticks([0, 50, 100])
plt.xlim(xlim)
plt.xlabel('Training Epoch')
plt.title(env_name)
plt.yticks([0,400,800], rotation='vertical')
plt.ylim(-10, 810)


plt.tight_layout()
plt.savefig('../out/cc_amn.pdf')

plt.show()

