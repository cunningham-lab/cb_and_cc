# The continuous Bernoulli: fixing a pervasive error in variational autoencoders

This repo contains example code (Tensorflow 1) from [The continuous Bernoulli: fixing a pervasive error in variational autoencoders](https://arxiv.org/abs/1907.06845). Different likelihoods are in separate notebooks for didactic purposes, cb_vae_mnist.ipynb implements the continuous Bernoulli VAE on MNIST. If you are only interested in the log normalizing constant of the continuous Bernoulli, see the code snippet below:
```
def cont_bern_log_norm(lam, l_lim=0.49, u_lim=0.51):
    # computes the log normalizing constant of a continuous Bernoulli distribution in a numerically stable way.
    # returns the log normalizing constant for lam in (0, l_lim) U (u_lim, 1) and a Taylor approximation in
    # [l_lim, u_lim].
    # cut_y below might appear useless, but it is important to not evaluate log_norm near 0.5 as tf.where evaluates
    # both options, regardless of the value of the condition.
    cut_lam = tf.where(tf.logical_or(tf.less(lam, l_lim), tf.greater(lam, u_lim)), lam, l_lim * tf.ones_like(lam))
    log_norm = tf.log(tf.abs(2.0 * tf.atanh(1 - 2.0 * cut_lam))) - tf.log(tf.abs(1 - 2.0 * cut_lam))
    taylor = tf.log(2.0) + 4.0 / 3.0 * tf.pow(lam - 0.5, 2) + 104.0 / 45.0 * tf.pow(lam - 0.5, 4)
    return tf.where(tf.logical_or(tf.less(lam, l_lim), tf.greater(lam, u_lim)), log_norm, taylor)
```
