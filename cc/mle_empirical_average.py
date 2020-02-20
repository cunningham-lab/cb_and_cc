import numpy as np
import pandas as pd
import dirichlet
from mcb_samplers import sample_mcb_naive_ordered
from cc_funcs import lambda_to_eta, cc_mean
import tensorflow as tf

sample_sizes = [2,5,10,20]

#------------------------------------------
# Set up parameters
#------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seedprior', dest='seedprior', type=int, default=0)
parser.add_argument('--seedsim', dest='seedsim', type=int, default=0)
parser.add_argument('--K', dest='K', type=int, default=3)
parser.add_argument('--trials', dest='trials', type=int, default=10000)

args = parser.parse_args()
K = args.K
seedprior = args.seedprior
seedsim = args.seedsim
trials = args.trials

results_dir = '../simulations/averageMLE_K{}_trials{}_seedprior{}_seedsim{}.pkl'.format(K, trials, seedprior, seedsim)

np.random.seed(seedprior)
tf.random.set_seed(seedprior)

# First set up the priors:
alpha_true = -np.log(np.random.rand(K))
Dir_true = alpha_true / sum(alpha_true)

lam_true = np.random.dirichlet(np.ones(K))
lam = lam_true.reshape(K, 1).transpose()
CC_true = cc_mean(lambda_to_eta(tf.convert_to_tensor(lam, dtype='float64')))[0].numpy()
while(np.any(np.logical_or(np.isnan(CC_true), CC_true<0))):
    lam_true = np.random.dirichlet(np.ones(K))
    lam = lam_true.reshape(K, 1).transpose()
    CC_true = cc_mean(lambda_to_eta(tf.convert_to_tensor(lam, dtype='float64')))[0].numpy()

results = pd.DataFrame({'Dir true': Dir_true, 'CC true': CC_true})
print(results)

# Now set the simulation seed
# We set this after the priorseed since we may want to
# use different simulation runs for the same prior
# to achieve increased parallelization
np.random.seed(seedsim)
tf.random.set_seed(seedsim)


def dir_ave_MLE(n):
    # Compute the empirical average of the Dirichlet and CC MLEs
    # for Dirichlet-generated data
    dir_means = []
    CC_means = []
    for i in range(trials):
        dat = np.random.dirichlet(alpha_true, n)

        # the CC MLE is just the empirical mean
        CC_means.append(dat.mean(axis=0))
        try:
            alpha_hat = dirichlet.mle(dat)
            mean = alpha_hat / sum(alpha_hat)
            dir_means.append(mean)
        except:
            print("WARNING: failed to converge")
    CC_means = np.array(CC_means)
    dir_means = np.array(dir_means)
    CC_mean = CC_means.mean(axis=0)
    dir_mean = dir_means.mean(axis=0)
    return CC_mean, dir_mean


def CC_ave_MLE(n):
    # Compute the empirical average of the Dirichlet and CC MLEs
    # for CC generated data
    dir_means = []
    CC_means = []
    lam = lam_true.repeat(n).reshape(K, n).transpose()

    for i in range(trials):
        dat = sample_mcb_naive_ordered(lam=lam)

        # the CC MLE is just the empirical mean
        CC_means.append(dat.mean(axis=0))
        try:
            alpha_hat = dirichlet.mle(dat)
            mean = alpha_hat / sum(alpha_hat)
            dir_means.append(mean)
        except:
            print("WARNING: failed to converge")
    CC_means = np.array(CC_means)
    dir_means = np.array(dir_means)
    CC_mean = CC_means.mean(axis=0)
    dir_mean = dir_means.mean(axis=0)
    return CC_mean, dir_mean

# Collect the results in a dataframe and save
# for collection by an aggregation script
for n in sample_sizes:
    CC, Dir = dir_ave_MLE(n)
    results["Dir data, CC MLE, n=" + str(n) ] = CC
    results["Dir data, Dir MLE, n=" + str(n)] = Dir
    CC, Dir = CC_ave_MLE(n)
    results["CC data, CC MLE, n=" + str(n)] = CC
    results["CC data, Dir MLE, n=" + str(n)] = Dir

results.to_pickle(results_dir)

print(results)
