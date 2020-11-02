"""Creates table showing output of our ablation study.

Requires that all our label-smoothing runs already be in place.
"""

import pandas as pd
import numpy as np

# Hyperparameters that are kept constant across the ablation
num_epochs = 500
optimizer = 'Adam'
lr = 0.001
alpha = 0.1

# Each run has a different combination of weight decay, dropout, & batchnorm
wds = [0.0001, 0.0001, 0.0, 0.0, 0.0001, 0.0001, 0.0, 0.0]
dps = ['', 'noDP'] * 4
bns = [''] * 4 + ['noBN'] * 4
notes = [x + y for x, y in zip(dps, bns)]

# Each combination was run under our 3 losses and for 10 seeds
losses = ['XE', 'LS', 'CC']
seeds = np.arange(0, 10)

res_means = []
res_stds = []
# Loop over everything and gather results
for i in range(len(wds)):
    means = []
    stds = []
    for loss in losses:
        max_vals = []
        for seed in seeds:
            results_path = "./out/cifar10_CNN_loss{}_opt{}_lr{}_wd{}_alpha{}_seed{}{}.csv".format(loss, optimizer, lr, wds[i], alpha, seed, notes[i])
            results = pd.read_csv(results_path)
            results = results[0:num_epochs] # In case we restrict training
            max_vals.append(results['val_accuracy'].max())
            # max_vals.append(results['val_accuracy'][-100:].mean())

        means.append(np.mean(max_vals))
        stds.append(np.std(max_vals))

    res_means.append(means)
    res_stds.append(stds)

# Put results into a dataframe and print out
model_desc = pd.DataFrame(np.array([dps, wds, bns]).T, columns=['DP', 'WD', 'BN'])
result_mean = pd.DataFrame(res_means, columns=losses)
result_stds = pd.DataFrame(res_stds, columns=losses)

print('Mean Scores:')
print(pd.concat([model_desc, result_mean], axis=1))
print('Stds:')
print(pd.concat([model_desc, result_stds], axis=1))
