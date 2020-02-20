# Fits GLM models to the UK election data

import tensorflow as tf
import tensorflow_probability as tfp
from cc_funcs import cc_log_prob, cc_mean, eta_to_lambda
from election_data import load_election_data
import numpy as np
import pandas as pd

seed = 0
epochs = 500
reg_cc = 0.0
reg_dir = 0.0
lr_cc = 0.5
lr_dir = 0.1
notes = ''

results_dir = '../out/glm_results_lrcc{}_lrdir{}_regcc{}_regdir{}_seed{}{}.pkl'.format(lr_cc, lr_dir, reg_cc, reg_dir, seed, notes)

tf.random.set_seed(seed)
tfd = tfp.distributions
num_parties = 4
df = load_election_data(num_parties)

class CC_GLM(object):
    def __init__(self, xdim, ydim, reg=0.0):
        self.xdim = xdim
        self.ydim = ydim
        self.regularization_strength = reg
        self.W = tf.Variable(tf.random.normal([self.xdim, self.ydim]))
        self.b = tf.Variable(tf.random.normal([self.ydim]))

    def __call__(self, X):
        X = tf.convert_to_tensor(X, dtype='float32')
        eta = self.b + tf.linalg.matmul(X, self.W)
        return eta

    def log_prob(self, X, Y):
        eta = self.__call__(X)
        return cc_log_prob(Y, eta)

    def mean(self, X):
        eta = self.__call__(X)
        return cc_mean(eta)

class dir_GLM(object):
    def __init__(self, xdim, ydim, reg=0.0):
        self.xdim = xdim
        self.ydim = ydim
        self.regularization_strength = reg
        self.W = tf.Variable(tf.random.normal([self.xdim, self.ydim]))
        self.b = tf.Variable(tf.random.normal([self.ydim]))

    def __call__(self, X):
        X = tf.convert_to_tensor(X, dtype='float32')
        log_alpha = self.b + tf.linalg.matmul(X, self.W)
        return tf.math.exp(log_alpha)

    def log_prob(self, X, Y):
        alpha = self.__call__(X)
        dist = tfd.Dirichlet(alpha)
        return dist.log_prob(Y)

    def mean(self, X):
        alpha = self.__call__(X)
        dist = tfd.Dirichlet(alpha)
        return dist.mean()

test_idx = (df['test'] == 1).values
Y_train = df.iloc[~test_idx, 0:num_parties+1].values
Y_test = df.iloc[test_idx, 0:num_parties+1].values
# Move Y away from zero
Y_train = Y_train + 1e-3 / (1 + 1e-3 * (num_parties))
Y_test = Y_test + 1e-3 / (1 + 1e-3 * (num_parties))
X_train = df.iloc[~test_idx, num_parties+1:-1].values
X_test = df.iloc[test_idx, num_parties+1:-1].values
num_predictors = X_test.shape[1]

cc_model = CC_GLM(num_predictors, num_parties, reg_cc)
dir_model = dir_GLM(num_predictors, num_parties+1, reg=reg_dir)
cc_optimizer = tf.keras.optimizers.Adam(lr_cc)
dir_optimizer = tf.keras.optimizers.Adam(lr_dir)
# optimizer = tf.keras.optimizers.RMSprop(lr_cc)
# dir_optimizer = tf.keras.optimizers.RMSprop(lr_dir)
# optimizer = tf.keras.optimizers.SGD(lr_cc)
# dir_optimizer = tf.keras.optimizers.SGD(lr_dir)
# optimizer = tf.keras.optimizers.Adadelta(lr_cc)
# dir_optimizer = tf.keras.optimizers.Adadelta(lr_dir)


# @tf.function
def compute_apply_gradients(model, X, Y, optimizer):
    with tf.GradientTape() as tape:
        prior = - tf.reduce_sum(tf.square(model.W)) - tf.reduce_sum(tf.square(model.b))
        negloglik = tf.reduce_sum(-model.log_prob(X, Y)) - model.regularization_strength * prior
    dW, db = tape.gradient(negloglik, [model.W, model.b])
    optimizer.apply_gradients(zip([dW, db], [model.W, model.b]))
    return - negloglik

loglik_cc = []
loglik_dir = []
l2_cc_err = []
l1_cc_err = []
l2_dir_err = []
l1_dir_err = []
print('Reg CC/Dir', reg_cc, reg_dir)
print('LR CC/Dir', lr_cc, lr_dir)
for epoch in range(1, epochs + 1):
    # At every iteration, we start by removing the observations that cause
    # numerical issues
    temp_mean = cc_model.mean(X_train)
    to_keep = tf.logical_not(tf.reduce_any(tf.math.is_nan(temp_mean), axis=1))
    X_cc_train = X_train[to_keep]
    Y_cc_train = Y_train[to_keep]
    print('epoch:', epoch)
    # start_time = time.time()
    obj = compute_apply_gradients(cc_model, X_cc_train, Y_cc_train, cc_optimizer)
    loglik_cc.append(obj)
    # end_time = time.time()

    temp_mean = cc_model.mean(X_test)
    to_keep = tf.logical_not(tf.reduce_any(tf.math.is_nan(temp_mean), axis=1))
    X_cc_test = X_test[to_keep]
    Y_cc_test = Y_test[to_keep]
    l2_cc_err.append(tf.reduce_mean(tf.sqrt(tf.losses.MSE(Y_cc_test,cc_model.mean(X_cc_test)))))
    l1_cc_err.append(tf.reduce_mean(tf.losses.MAE(Y_cc_test,cc_model.mean(X_cc_test))))
    print('CC loglik:', loglik_cc[-1])
    print('L2 CC test error:', l2_cc_err[-1])
    print('L1 CC test error:', l1_cc_err[-1])

    temp_loglik = dir_model.log_prob(X_train, Y_train)
    to_keep = tf.math.is_finite(temp_loglik)
    print(len(tf.where(to_keep)))
    X_dir_train = X_train[to_keep]
    Y_dir_train = Y_train[to_keep]
    # Also run gradient steps on the Dirichlet model
    obj = compute_apply_gradients(dir_model, X_dir_train, Y_dir_train, dir_optimizer)
    loglik_dir.append(obj)

    l2_dir_err.append(tf.reduce_mean(tf.sqrt(tf.losses.MSE(Y_test,dir_model.mean(X_test)))))
    l1_dir_err.append(tf.reduce_mean(tf.losses.MAE(Y_test,dir_model.mean(X_test))))
    print('Dir loglik:', loglik_dir[-1])
    print('L2 Dir test error:', l2_dir_err[-1])
    print('L1 Dir test error:', l1_dir_err[-1])

results = pd.DataFrame({'train_step': np.arange(epochs),
                        'CC_loglik': loglik_cc,
                        'Dir_loglik': loglik_dir,
                        'CC_L2_err': l2_cc_err,
                        'CC_L1_err': l1_cc_err,
                        'Dir_L2_err': l2_dir_err,
                        'Dir_L1_err': l1_dir_err})

# Save results:
results.to_pickle(results_dir)
