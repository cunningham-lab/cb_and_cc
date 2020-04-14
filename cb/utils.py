import numpy as np
import tensorflow as tf
import os
import struct
from sklearn.neighbors import KNeighborsClassifier
from six.moves import cPickle


class PermManager(object):
    # helper class for batching
    def __init__(self, n, batch_size, perm=None, perm_index=0, epoch=0):
        assert batch_size <= n
        self.n = n
        self.batch_size = batch_size
        if perm is None:
            self.perm = np.random.permutation(self.n)
        else:
            self.perm = perm
        self.perm_index = perm_index
        self.epoch = epoch

    def get_indices(self):
        if self.perm_index + self.batch_size < self.n:
            indices = self.perm[self.perm_index:self.perm_index + self.batch_size]
            self.perm_index = self.perm_index + self.batch_size
        elif self.perm_index + self.batch_size == self.n:
            indices = self.perm[self.perm_index:self.perm_index + self.batch_size]
            self.perm_index = 0
            self.perm = np.random.permutation(self.n)
            self.epoch = self.epoch + 1
        else:
            surplus = self.perm_index + self.batch_size - self.n
            indices1 = self.perm[self.perm_index:]
            self.perm = np.random.permutation(self.n)
            indices2 = self.perm[:surplus]
            indices = np.concatenate((indices1, indices2), axis=0)
            self.perm_index = surplus
            self.epoch = self.epoch + 1
        return indices


def read_mnist(dataset="training", path="."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return img, lbl


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def make_one_hot(labels, n_classes):
    n = np.shape(labels)[0]
    out = np.zeros([n, n_classes])
    for i in xrange(n):
        out[i, labels[i]] = 1.0
    return np.array(out, dtype='float32')


def warp(x, gamma):
    assert gamma >= -0.5 and gamma <= 0.5
    if gamma == -0.5:  # binarize
        res = np.where(x >= 0.5, 1.0, 0.0)
    elif gamma < 0:
        res = np.minimum(np.maximum(0.0, (x + gamma) / (1.0 + 2.0 * gamma)), 1.0)
    else:
        res = gamma + (1.0 - 2.0 * gamma) * x
    return res


def encoder_mnist(x, n_hidden, n_output, keep_prob):
    with tf.variable_scope("encoder"):
        hid = tf.layers.dense(x, n_hidden, activation=tf.nn.relu, name='fc1')
        hid = tf.nn.dropout(hid, keep_prob, name='drop1')
        hid = tf.layers.dense(hid, n_hidden, activation=tf.nn.relu, name='fc2')
        hid = tf.nn.dropout(hid, keep_prob, name='drop2')
        gaussian_params = tf.layers.dense(hid, 2 * n_output, activation=None, name='fc3')
        mean = gaussian_params[:, :n_output]
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])
    return mean, stddev


def encoder_cifar(x, n_output, feat=32):
    with tf.variable_scope("encoder"):
        hid = tf.reshape(x, [-1, 32, 32, 3])
        hid = tf.layers.conv2d(hid, 3, 2, padding='same', activation=tf.nn.relu, name='conv1')
        hid = tf.layers.conv2d(hid, feat, 2, padding='same', activation=tf.nn.relu, strides=2, name='conv2')
        hid = tf.layers.conv2d(hid, feat, 3, padding='same', activation=tf.nn.relu, name='conv3')
        hid = tf.layers.conv2d(hid, feat, 3, padding='same', activation=tf.nn.relu, name='conv4')
        hid = tf.contrib.layers.flatten(hid)
        hid = tf.layers.dense(hid, 128, activation=tf.nn.relu, name='fc1')
        gaussian_params = tf.layers.dense(hid, 2 * n_output, activation=None, name='fc2')
        mean = gaussian_params[:, :n_output]
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])
    return mean, stddev


def decoder_mnist_cb(z, n_hidden, n_output, keep_prob):
    with tf.variable_scope("decoder"):
        hid = tf.layers.dense(z, n_hidden, activation=tf.nn.relu, name='fc1')
        hid = tf.nn.dropout(hid, keep_prob, name='drop1')
        hid = tf.layers.dense(hid, n_hidden, activation=tf.nn.relu, name='fc2')
        hid = tf.nn.dropout(hid, keep_prob, name='drop2')
        lam = tf.layers.dense(hid, n_output, activation=tf.sigmoid, name='fc3')
    return lam


def decoder_mnist_norm(z, n_hidden, n_output, keep_prob):
    with tf.variable_scope("decoder"):
        hid = tf.layers.dense(z, n_hidden, activation=tf.nn.relu, name='fc1')
        hid = tf.nn.dropout(hid, keep_prob, name='drop1')
        hid = tf.layers.dense(hid, n_hidden, activation=tf.nn.relu, name='fc2')
        hid = tf.nn.dropout(hid, keep_prob, name='drop2')
        gaussian_params = tf.layers.dense(hid, 2 * n_output, activation=None, name='fc3')
        mean = gaussian_params[:, :n_output] # we found better performance without a sigmoid
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])
    return mean, stddev


def decoder_mnist_beta(z, n_hidden, n_output, keep_prob):
    with tf.variable_scope("decoder"):
        hid = tf.layers.dense(z, n_hidden, activation=tf.nn.relu, name='fc1')
        hid = tf.nn.dropout(hid, keep_prob, name='drop1')
        hid = tf.layers.dense(hid, n_hidden, activation=tf.nn.relu, name='fc2')
        hid = tf.nn.dropout(hid, keep_prob, name='drop2')
        beta_params = tf.layers.dense(hid, 2 * n_output, activation=None, name='fc3')
        alpha = 1e-6 + tf.nn.softplus(beta_params[:, :n_output])
        beta = 1e-6 + tf.nn.softplus(beta_params[:, n_output:])
    return alpha, beta


def decoder_cifar_cb(z, feat=32):
    with tf.variable_scope("decoder"):
        hid = tf.layers.dense(z, 128, activation=tf.nn.relu, name='fc1')
        hid = tf.layers.dense(hid, feat * 256, activation=tf.nn.relu, name='fc2')
        hid = tf.reshape(hid, [-1, 16, 16, feat])
        hid = tf.layers.conv2d_transpose(hid, feat, 3, padding='same', activation=tf.nn.relu, name='tconv1')
        hid = tf.layers.conv2d_transpose(hid, feat, 3, padding='same', activation=tf.nn.relu, name='tconv2')
        hid = tf.layers.conv2d_transpose(hid, feat, 3, padding='same', activation=tf.nn.relu, strides=2, name='tconv3')
        hid = tf.layers.conv2d(hid, 3, 2, padding='same', activation=tf.nn.sigmoid, name='conv1')
        lam = tf.reshape(hid, [tf.shape(hid)[0], -1])
    return lam


def decoder_cifar_norm(z, feat=32):
    with tf.variable_scope("decoder"):
        hid = tf.layers.dense(z, 128, activation=tf.nn.relu, name='fc1')
        hid = tf.layers.dense(hid, feat * 256, activation=tf.nn.relu, name='fc2')
        hid = tf.reshape(hid, [-1, 16, 16, feat])
        hid = tf.layers.conv2d_transpose(hid, feat, 3, padding='same', activation=tf.nn.relu, name='tconv1')
        hid = tf.layers.conv2d_transpose(hid, feat, 3, padding='same', activation=tf.nn.relu, name='tconv2')
        hid = tf.layers.conv2d_transpose(hid, feat, 3, padding='same', activation=tf.nn.relu, strides=2, name='tconv3')
        hid = tf.layers.conv2d(hid, 6, 2, padding='same', activation=None, name='conv1')
        mean = tf.sigmoid(hid[:, :, :, :3])
        mean = tf.reshape(mean, [tf.shape(mean)[0], -1])
        stddev = 1e-6 + tf.nn.softplus(hid[:, :, :, 3:])
        stddev = tf.reshape(stddev, [tf.shape(stddev)[0], -1])
    return mean, stddev


def decoder_cifar_beta(z, feat=32):
    with tf.variable_scope("decoder"):
        hid = tf.layers.dense(z, 128, activation=tf.nn.relu, name='fc1')
        hid = tf.layers.dense(hid, feat * 256, activation=tf.nn.relu, name='fc2')
        hid = tf.reshape(hid, [-1, 16, 16, feat])
        hid = tf.layers.conv2d_transpose(hid, feat, 3, padding='same', activation=tf.nn.relu, name='tconv1')
        hid = tf.layers.conv2d_transpose(hid, feat, 3, padding='same', activation=tf.nn.relu, name='tconv2')
        hid = tf.layers.conv2d_transpose(hid, feat, 3, padding='same', activation=tf.nn.relu, strides=2, name='tconv3')
        hid = tf.layers.conv2d(hid, 6, 2, padding='same', activation=None, name='conv1')
        alpha = 1e-6 + tf.nn.softplus(hid[:, :, :, :3])
        alpha = tf.reshape(alpha, [tf.shape(alpha)[0], -1])
        beta = 1e-6 + tf.nn.softplus(hid[:, :, :, 3:])
        beta = tf.reshape(beta, [tf.shape(beta)[0], -1])
    return alpha, beta


def classifier_mnist(x, n_hidden, n_output, keep_prob):
    with tf.variable_scope("logits"):
        hid = tf.layers.dense(x, n_hidden, activation=tf.nn.relu, name='fc1')
        hid = tf.nn.dropout(hid, keep_prob, name='drop1')
        hid = tf.layers.dense(hid, n_hidden, activation=tf.nn.relu, name='fc2')
        hid = tf.nn.dropout(hid, keep_prob, name='drop2')
        logits = tf.layers.dense(hid, n_output, activation=None, name='fc3')
    return logits


def classifier_cifar(x, keep_prob, feat=32):
    with tf.variable_scope("logits"):
        hid = tf.reshape(x, [-1, 32, 32, 3])
        hid = tf.layers.conv2d(hid, feat, 3, padding='same', activation=tf.nn.relu, name='firstconv')
        for reso in xrange(3):
            for resi in xrange(10):
                res_hid = tf.layers.conv2d(hid, feat * 2**reso, 3, padding='same', activation=tf.nn.relu,
                                           name='conv1_'+str(reso)+'_'+str(resi))
                res_hid = tf.layers.batch_normalization(res_hid, name='batchnorm1_'+str(reso)+'_'+str(resi))
                res_hid = tf.layers.conv2d(res_hid, feat * 2**reso, 3, padding='same', activation=None,
                                           name='conv2_'+str(reso)+'_'+str(resi))
                hid = hid + res_hid
                hid = tf.layers.batch_normalization(hid, name='batchnorm2_'+str(reso)+'_'+str(resi))
            hid = tf.layers.conv2d(hid, feat * 2 ** reso, 3, padding='same', activation=None, strides=2,
                                   name='convdown_'+str(reso))
            hid = tf.nn.dropout(hid, keep_prob, name='dropout_'+str(reso))
            if reso < 2:
                hid = tf.layers.conv2d(hid, feat * 2**(reso+1), 3, padding='same', activation=tf.nn.relu,
                                       name='lastconv_'+str(reso))
        hid = tf.contrib.layers.flatten(hid)
        logits = tf.layers.dense(hid, 10, activation=None, name='fc2')
    return logits


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

def mean_from_lam(lam):
    # given a numpy array of lambda parameters of a continuous Bernoulli, returns the corresponding means.
    # the function handles cases where lambda is outside [0,1] in a way that makes implementing an approximation
    # to the inverse easier in tensorflow
    # not very efficient as it is not vectorized
    lam_shape = np.shape(lam)
    reshaped_lam = np.reshape(lam, -1)
    means = np.zeros_like(reshaped_lam)
    for i in xrange(len(reshaped_lam)):
        if reshaped_lam[i] >= 1:
            means[i] = reshaped_lam[i]
        elif reshaped_lam[i] == 0.5:
            means[i] = 0.5
        elif reshaped_lam[i] > 0:
            means[i] = reshaped_lam[i] / (2.0 * reshaped_lam[i] - 1.0) + 1.0 / (2.0 * np.arctanh(1.0 - 2.0 *
                                                                                                 reshaped_lam[i]))
        else:
            means[i] = reshaped_lam[i]
    means = np.reshape(means, lam_shape)
    return means

def cont_bern_mean(lam, l_lim=0.49, u_lim=0.51):
    # continuous Bernoulli mean funtion in tensorflow
    # just like the normalizing constant, it is computed in a numerically stable way around 0.5
    cut_lam = tf.where(tf.logical_or(tf.less(lam, l_lim), tf.greater(lam, u_lim)), lam, l_lim * tf.ones_like(lam))
    mu = cut_lam / (2.0 * cut_lam - 1.0) + 1.0 / (2.0 * tf.atanh(1.0 - 2.0 * cut_lam))
    taylor = 0.5 + (lam - 0.5) / 3.0 + 16.0 / 45.0 * tf.pow(lam - 0.5, 3)
    return tf.where(tf.logical_or(tf.less(lam, l_lim), tf.greater(lam, u_lim)), mu, taylor)

xs = np.linspace(0.0, 1.0 + 1e-4, 100)
ys = mean_from_lam(xs)


def cont_bern_lam(mu, knots=np.array(xs, dtype='float32'), vals_at_knots=np.array(ys, dtype='float32')):
    # given the mean parameter mu of a continuous Bernoulli distribution, returns the canonical parameter lambda.
    # while mu can be analytically computed from lambda (by an invertible function), this cannot be analytically
    # inverted, and thus a linear interpolation (knots and vals at knots correspond to x and y values, respectively,
    # of the corresponding inverse function mu)
    slopes = (vals_at_knots[1:] - vals_at_knots[:-1]) / (knots[1:] - knots[:-1])
    intercepts = vals_at_knots[1:] - slopes * knots[1:]
    inv_slopes = 1.0 / slopes
    inv_intercepts = - intercepts / slopes

    dims = tf.shape(mu)
    reshaped_mu = tf.reshape(mu, [-1])  # [?]
    diff_1 = tf.expand_dims(reshaped_mu, 1) - tf.expand_dims(vals_at_knots[:-1], 0)  # [?, n_knots-1]
    diff_2 = tf.expand_dims(vals_at_knots[1:], 0) - tf.expand_dims(reshaped_mu, 1)  # [?, n_knots-1]
    masks = tf.where(tf.logical_and(tf.greater_equal(diff_1, 0.0), tf.greater(diff_2, 0.0)),
                     tf.ones([tf.shape(reshaped_mu)[0], np.shape(vals_at_knots)[0] - 1]),
                     tf.zeros([tf.shape(reshaped_mu)[0], np.shape(vals_at_knots)[0] - 1]))  # [?, n_knots-1]
    all_lines = tf.expand_dims(reshaped_mu, 1) * tf.expand_dims(inv_slopes, 0) + tf.expand_dims(inv_intercepts, 0)
    all_lines = masks * all_lines
    reshaped_lam = tf.reduce_sum(all_lines, 1)
    lam = tf.reshape(reshaped_lam, dims)
    return lam


def sample_cont_bern(lam):
    u = np.random.uniform(0, 1, np.shape(lam))
    samples = np.where(np.logical_and(lam > 0.499, lam < 0.501), u,
                       (np.log(u * (2.0 * lam - 1.0) + 1.0 - lam) - np.log(1.0 - lam)) / (np.log(lam) - np.log(1.0 - lam)))
    return samples


def sample_bern(param):
    return np.random.binomial(1, param)


def sample_norm(mu, sigma):
    return np.random.normal(mu, sigma)


def mean_from_params_beta(alpha, beta):
    return alpha / (alpha + beta)


def sample_beta(alpha, beta):
    return np.random.beta(alpha, beta)
