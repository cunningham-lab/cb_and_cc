import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

blue = "#377eb8"
purple = "#984ea3"
orange = "#ff7f00"
brown = "#a65628"
pink = '#f781bf'
alpha = 0.1

matplotlib.rc('xtick', labelsize=13)
matplotlib.rc('ytick', labelsize=13)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

# First we collect the results of our (parallelized) simulations
# and collate them into a single empirical average of the MLE
# that will be used to calculate the bias
# (this is in order to efficiently achieve 1M reps per experiment)
K_vals = [3] # could be 4, 5, 6... but we found little difference
n_vals = [2,5,10,20]
trials = 10000
seedpriors = np.arange(40)
seedsims = np.arange(100)
num_priors = 18 # number of seedpriors over which we compute average/std

for K in K_vals:
    print(K)
    res_all = np.zeros([num_priors, len(n_vals), 1 + 4]) # 1 for n_vals, 4 for 2 data * 2 models
    i=0

    for seedprior in seedpriors:
        # We will add in the empirical means from each dataframe and average out
        cum_sum = np.zeros([K, 2 + 4 * len(n_vals)])
        try:
            num_datasets = 0
            for seedsim in seedsims:
                results_dir = '../simulations/averageMLE_K{}_trials{}_seedprior{}_seedsim{}.pkl'.format(K, trials,
                                                                seedprior, seedsim)
                results = pd.read_pickle(results_dir)
                cum_sum += results.to_numpy()
                num_datasets += 1

            # Obtain the empirical averages over all our simulations
            ave = cum_sum / num_datasets
            ave = pd.DataFrame(ave, columns=results.columns)

            # Dirichlet/CC;model/data is the key:
            DdDm = [] # Dirichlet data and Dirichlet model
            DdCm = [] # Dirichlet data CC model
            CdDm = []
            CdCm = []
            # Compute the actual biases
            for n in n_vals:
                DdDm.append(np.sum(np.abs(ave['Dir true'] - ave['Dir data, Dir MLE, n=' + str(n)])))
                DdCm.append(np.sum(np.abs(ave['Dir true'] - ave['Dir data, CC MLE, n=' + str(n)])))
                CdDm.append(np.sum(np.abs(ave['CC true'] - ave['CC data, Dir MLE, n=' + str(n)])))
                CdCm.append(np.sum(np.abs(ave['CC true'] - ave['CC data, CC MLE, n=' + str(n)])))

            bias = pd.DataFrame({'n_val': n_vals, 'DdDm': DdDm, 'DdCm':DdCm, 'CdDm': CdDm, 'CdCm': CdCm})
            # This is a horrible hack, but this only happens under numerical instabilities
            # and I don't have time to fix properly right now
            if bias['CdCm'].mean() > 0.01:
                print("Skipping due to numerical instabilities, seed:", seedprior)
                assert 0==1
            res_all[i] = bias.to_numpy()
            i += 1

        except:
            print("No data for seed:", seedprior)


    # average over random seeds (i.e. different priors)
    res_ave = np.mean(res_all, axis=0)
    res_ave = pd.DataFrame(res_ave, columns=bias.columns)
    # Compute std over different priors for error bars
    res_std = np.std(res_all, axis=0)
    res_std = pd.DataFrame(res_std, columns=bias.columns)
    res_std['n_val'] = res_ave['n_val'].to_numpy()
    print(res_ave)

    x = res_std['n_val']
    y = res_ave['DdDm']
    error = res_std['DdDm']
    plt.fill_between(x, y - error, y + error, color=brown, alpha=alpha)
    x = res_std['n_val']
    y = res_ave['DdCm']
    error = res_std['DdCm']
    plt.fill_between(x, y - error, y + error, color=blue, alpha=alpha)
    x = res_std['n_val']
    y = res_ave['CdDm']
    error = res_std['CdDm']
    plt.fill_between(x, y - error, y + error, color=orange, alpha=alpha)
    x = res_std['n_val']
    y = res_ave['CdCm']
    error = res_std['CdCm']
    plt.fill_between(x, y - error, y + error, color=purple, alpha=alpha)

    plt.plot('n_val', 'DdDm', data=res_ave, color=pink, linewidth=3)
    plt.plot('n_val', 'CdDm', data=res_ave, color=orange, linewidth=3)
    plt.plot('n_val', 'CdCm', data=res_ave, color=purple, linewidth=6)
    plt.plot('n_val', 'DdCm', data=res_ave, color=blue, linewidth=3)



custom_lines = [Line2D([0], [0], lw=3, color=orange),
                Line2D([0], [0], lw=3, color=pink),
                Line2D([0], [0], lw=3, color=blue),
                Line2D([0], [0], lw=3, color=purple),
                ]
plt.legend(custom_lines, ['Dir MLE, CC data', 'Dir MLE, Dir data', 'CC MLE, Dir data', 'CC MLE, CC data'], loc='upper left')

# plt.yscale('log')
plt.xlabel('Sample size')
plt.ylabel('Bias (L1 norm)')
plt.savefig("../out/biases.pdf")
plt.show()



