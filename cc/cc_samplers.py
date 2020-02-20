import numpy as np

# Define functions for naive rejection sampling of the MCB:

def sample_cont_bern(lam):
    # samples fom the CB distribution.
    # lam can have any shape and the output will have the same shape, correspoding to independent draws from
    # CB distributions.
    u = np.random.uniform(0, 1, np.shape(lam))
    samples = np.where(np.logical_and(lam > 0.499, lam < 0.501), u,
                       (np.log(u * (2.0 * lam - 1.0) + 1.0 - lam) - np.log(1.0 - lam)) / (
                                   np.log(lam) - np.log(1.0 - lam)))
    return samples


def sample_cc_naive_simple(lam, return_acceptance_rate=False):
    # uses the naive rejection sampler for the CC.
    # lam should be a 1-dimensional array corresponding to the parameter of one CC distribution, this
    # function does not sample from several different CCs at the same time.
    ind_lams = lam[:-1] / (lam[:-1] + lam[-1])
    not_accepted = True
    attempts = 0.0
    while not_accepted:
        if attempts > 1e5:
            not_accepted = False
            acc_rate = 1.0 / 1e5
        attempts += 1.0
        proposed_samples = sample_cont_bern(ind_lams)
        if np.sum(proposed_samples) <= 1.0:
            not_accepted = False
            acc_rate = 1.0 / attempts
    if return_acceptance_rate:
        return proposed_samples, acc_rate
    else:
        return proposed_samples


def sample_cc_naive(lam, return_acceptance_rates=False):
    # returns n samples from a K-1 dimensional CC using the naive rejection sampler.
    # lambda should have shape [n, K]
    # note: there might be a better way of getting n samples from sample_cc_naive_simple than simply using a for
    # loop, although I am not sure how to vectorize the while loop inside sample_cc_naive_simple.
    n, K = np.shape(lam)
    samples = np.zeros([n, K - 1])
    acc_rates = np.zeros(n)
    for i in range(n):
        if return_acceptance_rates:
            samples[i], acc_rates[i] = sample_cc_naive_simple(lam[i], True)
        else:
            samples[i] = sample_cc_naive_simple(lam[i])
    if return_acceptance_rates:
        return samples, acc_rates
    else:
        return samples


# Define functions for permutation-based rejection sampling of the CC:

def g_func(eta, Q_inv):
    return np.prod(np.power(eta, np.transpose(Q_inv)), axis=1)


def create_Omega_to_S_id_mat(size, return_inverse=True):
    # returns the matrix mapping the probability simplex Omega to S_id.
    # if return_inverse is true, also returns the inverse of this matrix
    # note: I'm sure there's a smarter way to implement this than to use a for loop, although for a fixed
    # size this computation can be only cariied out once, so being inefficient here doesn't really matter
    mat = np.ones([size, size])
    mat_inv = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            if i + j < size - 1:
                mat[i, j] = 0.0
            if j + i == size - 2:
                mat_inv[i, j] = -1.0
            if j + i == size - 1:
                mat_inv[i, j] = 1.0
    if return_inverse:
        return mat, mat_inv
    else:
        return mat


def compute_log_k(eta_1, eta_2):
    # maximizes k over S_id by checking every vertex.
    # while this is already efficient, there might be an even better way of doing this, it's probably
    # worth thinking about
    s = 0.0
    m = s
    for i in range(eta_1.size - 1, -1, -1):
        s += np.log(eta_1[i]) - np.log(eta_2[i])
        if s > m:
            m = s
    return m


def sample_cc_perm_simple(lam, return_acceptance_rate=False):
    # uses the permutation-based rejection sampler for the CC.
    # lam should be a 1-dimensional array corresponding to the parameter of one CC distribution, this
    # function does not sample from several different CCs at the same time.
    eta = lam[:-1] / lam[-1]
    _, B_inv = create_Omega_to_S_id_mat(eta.size)
    eta_tilde = g_func(eta, B_inv)
    # ordering_perm = np.argsort(eta_tilde)
    # eta_tilde = eta_tilde[ordering_perm]
    lam_tilde = np.concatenate([eta_tilde, [1.0]])
    ind_lam_tilde = lam_tilde[:-1] / (lam_tilde[:-1] + lam_tilde[-1])
    not_accepted = True
    attempts = 0.0
    while not_accepted:
        if attempts > 1e5:
            not_accepted = False
            acc_rate = 1.0 / 1e5
        attempts += 1.0
        proposed_sample = sample_cont_bern(ind_lam_tilde)
        sigma = np.argsort(proposed_sample)
        sigma_inv = np.empty(sigma.size, sigma.dtype)
        sigma_inv[sigma] = np.arange(sigma.size)
        proposed_sample = proposed_sample[sigma]
        perm_mat_inv = np.eye(sigma_inv.size)[sigma_inv]
        eta_proposal = g_func(eta_tilde, perm_mat_inv)
        log_k = compute_log_k(eta_tilde, eta_proposal)
        u = np.random.uniform(0, 1)
        log_alpha = np.sum(proposed_sample * (np.log(eta_tilde) - np.log(eta_proposal))) - log_k
        if np.log(u) < log_alpha:
            not_accepted = False
            acc_rate = 1.0 / attempts
    sample = np.matmul(B_inv, proposed_sample)  # TODO: this can be made much more efficiently
    # ordering_perm_inv = np.empty(ordering_perm.size, ordering_perm.dtype)
    # ordering_perm_inv[ordering_perm] = np.arange(ordering_perm.size)
    # sample = sample[ordering_perm_inv]
    if return_acceptance_rate:
        return sample, acc_rate
    else:
        return sample


def sample_cc_perm(lam, return_acceptance_rates=False):
    # returns n samples from a K-1 dimensional CC using the permutation-based rejection sampler.
    # lambda should have shape [n, K-1]
    # note: there might be a better way of getting n samples from sample_cc_perm_simple than simply using a for
    # loop, although I am not sure how to vectorize the while loop inside sample_cc_perm_simple.
    n, K = np.shape(lam)
    samples = np.zeros([n, K - 1])
    acc_rates = np.zeros(n)
    for i in range(n):
        if return_acceptance_rates:
            samples[i], acc_rates[i] = sample_cc_perm_simple(lam[i], True)
        else:
            samples[i] = sample_cc_perm_simple(lam[i])
    if return_acceptance_rates:
        return samples, acc_rates
    else:
        return samples

# Define functions for ordered rejection sampling of the CC:

# Inverse CDF of continuous bernoulli
def inv_cdf(u, l):
    if l > 0.499 and l < 0.501:
        return u
    else:
        num = np.log(u*(2*l-1) + 1 - l) - np.log(1-l)
        den = np.log(l) - np.log(1-l)
        return num / den

# ordered rejection sampler
def sample_cc_ordered_simple(lam, return_acceptance_rate=False, return_unif=False):
    l = np.array(lam)
    dim = len(l)
    l_sort = l.argsort()[::-1]
    l_sort_inv = np.empty(l_sort.size, l_sort.dtype)
    l_sort_inv[l_sort] = np.arange(l_sort.size)
    l.sort()
    l = l[::-1]
    not_accepted = True
    sample = np.zeros(dim)
    attempts = 0.0
    while not_accepted:
        cum_sum = 0.0
        U = np.random.rand(dim)
        for j in range(1,dim):
            attempts += 1
            sample[j] = inv_cdf(U[j], l[j]/(l[j] + l[0]))
            cum_sum += sample[j]
            if cum_sum > 1:
                break
        if cum_sum < 1:
            not_accepted = False
        if attempts > 1e5:
            not_accepted = False
            sample = sample * np.nan
    sample[0] = 1 - sample.sum()
    if return_unif:
        sample = U
        sample[0] = np.nan
    sample = sample[l_sort_inv]
    if return_acceptance_rate:
        return sample, 1.0 / attempts * (dim - 1)
    else:
        return sample

def sample_cc_ordered(lam, return_acceptance_rates=False, return_unif=False):
    # returns n samples from a K-1 dimensional CC using the permutation-based rejection sampler.
    # lambda should have shape [n, K]
    n, K = np.shape(lam)
    samples = np.zeros([n, K])
    acc_rates = np.zeros(n)
    for i in range(n):
        if return_acceptance_rates:
            samples[i], acc_rates[i] = sample_cc_ordered_simple(lam[i], True, return_unif)
        else:
            samples[i] = sample_cc_ordered_simple(lam[i], return_unif=return_unif)
    if return_acceptance_rates:
        return samples, acc_rates
    else:
        return samples
