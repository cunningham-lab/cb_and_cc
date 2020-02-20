# Fits neural networks to the UK election data

import tensorflow as tf
import tensorflow_probability as tfp
from cc_funcs import cc_log_prob, cc_mean
from election_data import load_election_data
import numpy as np
import pandas as pd

seed = 0
epochs = 1000
TRAIN_BUF = 509
TEST_BUF = 150
BATCH_SIZE = 128
reg_cc = 0.0
reg_dir = 0.0
lr_cc = 0.01
lr_dir = 0.01
dp_rate = 0.0
notes = ''

results_dir = '../out/mlp_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}_dp{}{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, dp_rate, notes)

tf.random.set_seed(seed)
tfd = tfp.distributions
num_parties = 4
df = load_election_data(num_parties)

cc_optimizer = tf.keras.optimizers.Adam(lr_cc)
dir_optimizer = tf.keras.optimizers.Adam(lr_dir)
# optimizer = tf.keras.optimizers.RMSprop(lr_cc)
# dir_optimizer = tf.keras.optimizers.RMSprop(lr_dir)
# optimizer = tf.keras.optimizers.SGD(lr_cc)
# dir_optimizer = tf.keras.optimizers.SGD(lr_dir)
# optimizer = tf.keras.optimizers.Adadelta(lr_cc)
# dir_optimizer = tf.keras.optimizers.Adadelta(lr_dir)

class CC_MLP(tf.keras.Model):
    def __init__(self, xdim, ydim, reg=0.0):
        super(CC_MLP, self).__init__()
        self.xdim = xdim
        self.ydim = ydim
        self.regularization_strength = reg
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.xdim,)),
                # tf.keras.layers.Dense(20, activation='relu'),
                tf.keras.layers.Dense(20, activation='relu'),
                # tf.keras.layers.Dropout(dp_rate),
                tf.keras.layers.Dense(self.ydim),
            ]
        )

    def call(self, X):
        X = tf.convert_to_tensor(X, dtype='float32')
        eta = self.net(X)
        return eta

    def compute_loglik(self, X, Y):
        eta = self.call(X)
        return cc_log_prob(Y, eta)

    def mean(self, X):
        eta = self.call(X)
        return cc_mean(eta)


class Dir_MLP(tf.keras.Model):
    def __init__(self, xdim, ydim, reg=0.0):
        super(Dir_MLP, self).__init__()
        self.xdim = xdim
        self.ydim = ydim
        self.regularization_strength = reg
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.xdim,)),
                tf.keras.layers.Dense(20, activation='relu'),
                # tf.keras.layers.Dropout(dp_rate),
                tf.keras.layers.Dense(self.ydim),
            ]
        )

    def call(self, X):
        X = tf.convert_to_tensor(X, dtype='float32')
        log_alpha = self.net(X)
        return tf.math.exp(log_alpha)

    def compute_loglik(self, X, Y):
        alpha = self.call(X)
        dist = tfd.Dirichlet(alpha)
        return dist.log_prob(Y)

    def mean(self, X):
        alpha = self.call(X)
        dist = tfd.Dirichlet(alpha)
        return dist.mean()


test_idx = (df['test'] == 1).values
Y_train = df.iloc[~test_idx, 0:num_parties+1].values
Y_test = df.iloc[test_idx, 0:num_parties+1].values
X_train = df.iloc[~test_idx, num_parties+1:-1].values
X_test = df.iloc[test_idx, num_parties+1:-1].values
num_predictors = X_test.shape[1]
Y_train = Y_train + 1e-3 / (1 + 1e-3 * (num_parties))
Y_test = Y_test + 1e-3 / (1 + 1e-3 * (num_parties))

X_train = tf.convert_to_tensor(X_train, dtype='float32')
X_test = tf.convert_to_tensor(X_test, dtype='float32')

Y_train = tf.convert_to_tensor(Y_train, dtype='float32')
Y_test = tf.convert_to_tensor(Y_test, dtype='float32')

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).shuffle(TEST_BUF).batch(BATCH_SIZE)

cc_model = CC_MLP(num_predictors, num_parties, reg_cc)
dir_model = Dir_MLP(num_predictors, num_parties+1, reg=reg_dir)

# @tf.function
def compute_apply_gradients(model, X, Y, optimizer):
    with tf.GradientTape() as tape:
        prior = - sum(tf.reduce_sum(tf.square(w)) for w in model.trainable_variables)
        negloglik = tf.reduce_sum(-model.compute_loglik(X, Y)) - model.regularization_strength * prior
    gradients = tape.gradient(negloglik, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return -negloglik

loglik_cc = []
loglik_dir = []
l2_cc_err = []
l1_cc_err = []
l2_dir_err = []
l1_dir_err = []
print('Reg CC/Dir', reg_cc, reg_dir)
print('LR CC/Dir', lr_cc, lr_dir)
Y_test_dir = Y_test + 1e-3 / (1 + Y_test.shape[1] * 1e-3)
for epoch in range(1, epochs + 1):
    print('epoch:', epoch)

    for train_x, train_y in train_dataset:
        # At every iteration, we start by removing the observations that cause
        # numerical issues
        temp_mean = cc_model.mean(train_x)
        to_keep = tf.logical_not(tf.reduce_any(tf.logical_or(temp_mean < 0, tf.math.is_nan(temp_mean)), axis=1))
        X_cc_train = train_x[to_keep]
        Y_cc_train = train_y[to_keep]
        obj = compute_apply_gradients(cc_model, X_cc_train, Y_cc_train, cc_optimizer)

        X_dir_train = train_x
        Y_dir_train = train_y
        obj = compute_apply_gradients(dir_model, X_dir_train, Y_dir_train, dir_optimizer)

    temp_mean = cc_model.mean(X_test)
    to_keep = tf.logical_not(tf.reduce_any(tf.logical_or(temp_mean < 0, tf.math.is_nan(temp_mean)), axis=1))
    X_cc_test = X_test[to_keep]
    Y_cc_test = Y_test_dir[to_keep]
    l2_cc_err.append(tf.reduce_mean(tf.sqrt(tf.losses.MSE(Y_cc_test,cc_model.mean(X_cc_test)))))
    l1_cc_err.append(tf.reduce_mean(tf.losses.MAE(Y_cc_test,cc_model.mean(X_cc_test))))
    print('L2 CC test error:', l2_cc_err[-1])
    print('L1 CC test error:', l1_cc_err[-1])
    print('CC train err:', tf.reduce_mean(tf.sqrt(tf.losses.MSE(Y_train,cc_model.mean(X_train)))))
    temp_mean = cc_model.mean(X_train)
    to_keep = tf.logical_not(tf.reduce_any(tf.logical_or(temp_mean < 0, tf.math.is_nan(temp_mean)), axis=1))
    X_cc_train = X_train[to_keep]
    Y_cc_train = Y_train[to_keep]
    loglik = tf.reduce_sum(cc_model.compute_loglik(X_cc_train, Y_cc_train))
    loglik_cc.append(loglik)

    l2_dir_err.append(tf.reduce_mean(tf.sqrt(tf.losses.MSE(Y_test,dir_model.mean(X_test)))))
    l1_dir_err.append(tf.reduce_mean(tf.losses.MAE(Y_test,dir_model.mean(X_test))))
    print('L2 Dir test error:', l2_dir_err[-1])
    print('L1 Dir test error:', l1_dir_err[-1])
    print('Dir train err:', tf.reduce_mean(tf.sqrt(tf.losses.MSE(Y_train,dir_model.mean(X_train)))))
    # print('alpha:', dir_model(X_test))
    loglik = tf.reduce_sum(dir_model.compute_loglik(X_train, Y_train))
    loglik_dir.append(loglik)


results = pd.DataFrame({'train_step': np.arange(epochs),
                        'loglik_cc': loglik_cc,
                        'loglik_dir': loglik_dir,
                        'CC_L2_err': l2_cc_err,
                        'CC_L1_err': l1_cc_err,
                        'Dir_L2_err': l2_dir_err,
                        'Dir_L1_err': l1_dir_err})

# Save results:

results.to_pickle(results_dir)
