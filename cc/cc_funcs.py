import tensorflow as tf
import numpy as np


def lambda_to_eta(lam):
    eta = tf.math.log(lam / lam[:, -1, np.newaxis])
    return eta[:, 0:-1]

def eta_to_lambda(eta):
    n = eta.shape[0]
    aug_eta = tf.concat([eta, np.repeat(0.0, n)[:, np.newaxis]], -1)
    return tf.math.softmax(aug_eta)


def cc_log_normalizer(eta):
    # Compute the log-normalizer of the CC distribution
    # See paper/supplement for mathematical details
    n, K = eta.shape
    dtype = eta.dtype
    aug_eta = tf.concat([eta, tf.zeros([n,1], dtype=dtype)], -1)
    lse = tf.math.reduce_logsumexp(aug_eta, axis=1)
    res = - cc_log_norm_const(eta) + lse
    return res

def cc_log_norm_const(eta):
    # Compute the log-normalizing-constant of the CC distribution
    # See paper/supplement for mathematical details
    n, K = eta.shape
    dtype = eta.dtype
    eta_np = tf.stop_gradient(eta)
    sds = tf.math.reduce_std(eta_np, axis=1)
    scaling_factors = sds / 1.0
    aug_eta = tf.concat([eta, tf.zeros([n,1], dtype=dtype)], -1)
    lam = tf.math.softmax(aug_eta, axis=1)
    aug_eta = aug_eta / tf.reshape(scaling_factors, [n,1])
    rows = tf.reshape(aug_eta, [n, -1, 1])
    cols = tf.reshape(aug_eta, [n, -1, K+1])
    eta_diffs = rows - cols
    eta_diffs = eta_diffs + tf.eye(K+1, dtype=dtype)
    dens = tf.reduce_prod(eta_diffs, axis=1)
    res = - tf.math.log((-1)**K * tf.reduce_sum(lam / dens, axis=1)) +\
        K * tf.math.log(scaling_factors)
    return res

def cc_mean(eta):
    # Compute mean of CC variate by running autodiff on log-normalizer
    with tf.GradientTape() as tape:
        arg = tf.Variable(eta)
        C = cc_log_normalizer(arg)
    res = tape.gradient(C, arg)
    remainder = 1. - tf.reduce_sum(res, axis=1)
    res = tf.concat([res, remainder[:, np.newaxis]], axis=1)
    # Remove numerically unstable gradients
    bad_means = tf.reduce_any(res <= 0, axis=1)
    updates = res[bad_means] * np.nan
    res = tf.tensor_scatter_nd_update(res, tf.where(bad_means), updates)
    return res

def cc_log_prob(sample, eta):
    # Compute the likelihood term
    # loglik = tf.reduce_sum(sample[:, 0:-1] * eta, axis=1)
    n, K = eta.shape
    dtype = eta.dtype
    aug_eta = tf.concat([eta, tf.zeros([n,1], dtype=dtype)], -1)
    # lam = tf.math.softmax(aug_eta, axis=1)
    loglik = - tf.nn.softmax_cross_entropy_with_logits(sample, aug_eta)
    log_norm_const = cc_log_norm_const(eta)
    return loglik + log_norm_const
