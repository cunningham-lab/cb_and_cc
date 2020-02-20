import numpy as np
import time
import matplotlib.pyplot as plt

# Import our samplers
from cc_samplers import sample_cc_naive
from cc_samplers import sample_cc_perm
from cc_samplers import sample_cc_ordered

np.random.seed(0)

# Run a few experiments and summarize the results

K_vals = list(range(3,11))
n = 100
bins = [1,10,100,1000,10000,99900,1000000]
fig, axs = plt.subplots(3, len(K_vals))
for K in K_vals:
    print("K = " + str(K))
    lam_param = np.ones(K) / K  # note: we should play around with this
    lam = np.random.dirichlet(lam_param, n)
    start = time.time()
    ordered_samples, acceptance_rates = sample_cc_ordered(lam, return_acceptance_rates=True)
    end = time.time()
    print('done with ordered sampling')
    print('time taken:', end - start)
    print('average acceptance rate:', acceptance_rates.mean())
    axs[0, K-K_vals[0]].hist(np.log10(1 / acceptance_rates), bins = np.log10(bins))
    axs[0, K-K_vals[0]].set(ylim = [0,n], title='$K=$'+str(K), ylabel='ordered')
    start = time.time()
    perm_samples, acceptance_rates = sample_cc_perm(lam, return_acceptance_rates=True)
    end = time.time()
    print('done with permutation-based sampling')
    print('time taken:', end - start)
    print('average acceptance rate:', acceptance_rates.mean())
    axs[1, K-K_vals[0]].hist(np.log10(1 / acceptance_rates), bins = np.log10(bins))
    axs[1, K-K_vals[0]].set(ylim = [0,n], ylabel='permutation')
    start = time.time()
    naive_samples, acceptance_rates = sample_cc_naive(lam, return_acceptance_rates=True)
    end = time.time()
    print('done with naive sampling')
    print('time taken:', end - start)
    print('average acceptance rate:', acceptance_rates.mean())
    axs[2, K-K_vals[0]].hist(np.log10(1 / acceptance_rates), bins = np.log10(bins))
    axs[2, K - K_vals[0]].set(ylim = [0,n], ylabel='naive')

for ax in axs.flat:
    ax.label_outer()

fig.text(0.5, 0.03, '$\log_{10}$(proposals)', ha='center')
plt.xlabel(" ")
plt.tight_layout()

plt.savefig('../out/rejection_rates_hist.pdf')
plt.show()