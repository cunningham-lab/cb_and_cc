import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from cc_funcs import cc_log_prob, cc_mean
import numpy as np
import pandas as pd
from mnist_teacher import hinton
import time

seed = 0
epochs = 1000
reg_cc = 0.0
reg_dir = 0.0
reg_xe = 0.0
lr_cc = 0.1
lr_dir = 0.01
lr_xe = 0.1
dp_rate = 0.0
soft_loss = 1.0
hard_loss = 0.0
BATCH_SIZE = 1000
notes = ''
epochs_teacher = 200
path_teacher = '../tf_models/mnist_teacher_epochs{}'.format(epochs_teacher)
t_adj = 1.0
omit3 = False

results_dir = '../out/mnist_student_temp{}_omit3{}_lrcc{}_lrdir{}_lrxe{}_soft{}_hard{}_batch{}{}.pkl'.format(
    t_adj, omit3, lr_cc, lr_dir, lr_xe, soft_loss, hard_loss, BATCH_SIZE, notes)

TRAIN_BUF = 60000
TEST_BUF = 10000
print('Reg', reg_xe, reg_cc, reg_dir)
print('LR', lr_xe, lr_cc, lr_dir)

tf.random.set_seed(seed)
tfd = tfp.distributions
teacher = hinton(dp_rate)
teacher.load_weights(path_teacher)
teacher.temperature = t_adj

# Generate the soft targets
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizing the images to the range of [0., 1.]
train_images = train_images.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.

if omit3:
    to_keep = train_labels != 3
    train_images = train_images[to_keep]
    train_labels = train_labels[to_keep]


X_train = tf.convert_to_tensor(train_images, dtype='float32')
X_test = tf.convert_to_tensor(test_images, dtype='float32')

Y_train = teacher.call(X_train)
Y_test = teacher.call(X_test)

hard_train = tf.convert_to_tensor(train_labels, dtype='float32')
hard_test = tf.convert_to_tensor(test_labels, dtype='float32')

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train, hard_train)).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test, hard_test)).shuffle(TEST_BUF).batch(BATCH_SIZE)


class CC_MLP(tf.keras.Model):
    def __init__(self, ydim, reg=0.0):
        super(CC_MLP, self).__init__()
        self.ydim = ydim
        self.regularization_strength = reg
        self.net = tf.keras.Sequential(
            [
                keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(30, activation='relu'),
                tf.keras.layers.Dense(self.ydim),
            ]
        )

    # @tf.function
    def call(self, X):
        X = tf.convert_to_tensor(X, dtype='float32')
        eta = self.net(X)
        return eta

    # @tf.function
    def compute_loglik(self, X, Y):
        eta = self.call(X)
        temp_mean = tf.stop_gradient(cc_mean(eta))
        discard = tf.math.reduce_any(tf.math.is_nan(temp_mean), axis=1)
        keep = tf.logical_not(discard)
        # Y_reduced = Y[:, 0:-1]
        n = eta.shape[0]
        logits = tf.concat([eta, tf.zeros([n,1])], -1)
        # partial = tf.reduce_sum(logits[discard] * Y[discard], axis=1)
        partial = - tf.nn.softmax_cross_entropy_with_logits(Y[discard], logits[discard])
        full = cc_log_prob(Y[keep], eta[keep])
        return tf.reduce_sum(partial) + tf.reduce_sum(full)

    def compute_XE(self, X, Y):
        n = X.shape[0]
        eta = self.call(X)
        aug_eta = tf.concat([eta, tf.zeros([n,1])], -1)
        probhat = tf.math.softmax(aug_eta)
        return - tf.losses.sparse_categorical_crossentropy(Y, probhat)

    def mean(self, X):
        eta = self.call(X)
        return cc_mean(eta)

class XE_MLP(tf.keras.Model):
    def __init__(self, ydim, reg=0.0):
        super(XE_MLP, self).__init__()
        self.ydim = ydim
        self.regularization_strength = reg
        self.net = tf.keras.Sequential(
            [
                keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(30, activation='relu'),
                tf.keras.layers.Dense(self.ydim),
            ]
        )

    def call(self, X):
        n = X.shape[0]
        X = tf.convert_to_tensor(X, dtype='float32')
        logits = self.net(X)
        logits = tf.concat([logits, tf.zeros([n,1])], -1)
        return logits

    def compute_loglik(self, X, Y):
        logits = self.call(X)
        return - tf.nn.softmax_cross_entropy_with_logits(Y, logits)

    def compute_XE(self, X, Y):
        logits = self.call(X)
        probhat = tf.math.softmax(logits)
        return - tf.losses.sparse_categorical_crossentropy(Y, probhat)

class Dir_MLP(tf.keras.Model):
    def __init__(self, ydim, reg=0.0):
        super(Dir_MLP, self).__init__()
        self.ydim = ydim
        self.regularization_strength = reg
        self.net = tf.keras.Sequential(
            [
                keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(30, activation='relu'),
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

    def compute_XE(self, X, Y):
        alpha = self.call(X)
        prob_hat = alpha / tf.reshape(tf.reduce_sum(alpha, axis=1), [-1, 1])
        return - tf.losses.sparse_categorical_crossentropy(Y, prob_hat)



cc_model = CC_MLP(9, reg_cc)
xe_model = XE_MLP(9, reg_xe)
dir_model = Dir_MLP(10, reg=reg_dir)
optimizer = tf.keras.optimizers.Adam(lr_cc)
xe_optimizer = tf.keras.optimizers.Adam(lr_xe)
dir_optimizer = tf.keras.optimizers.Adam(lr_dir)

# We start the XE and CC model in the same place to make the comparison more fair
xe_model.set_weights(cc_model.get_weights())

# We try standard normal initializations as well
def initialize_weights(model):
    temp_weights = model.get_weights()
    for i in range(len(temp_weights)):
        temp_weights[i] = tf.random.normal(shape=temp_weights[i].shape)
    model.set_weights(temp_weights)
# initialize_weights(cc_model)
# initialize_weights(xe_model)
# initialize_weights(dir_model)

# @tf.function
def compute_apply_gradients(model, X, Y_soft, Y_hard, optimizer):
    with tf.GradientTape() as tape:
        prior = - sum(tf.reduce_sum(tf.square(w)) for w in model.trainable_variables)
        negloglik = soft_loss * tf.reduce_sum(-model.compute_loglik(X, Y_soft)) \
                    + hard_loss * tf.reduce_sum(-model.compute_XE(X, Y_hard)) \
                    - model.regularization_strength * prior
    gradients = tape.gradient(negloglik, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return -negloglik

loglik_cc = []
loglik_dir = []
loglik_xe = []
l2_cc_err = []
l1_cc_err = []
l2_xe_err = []
l1_xe_err = []
l2_dir_err = []
l1_dir_err = []
num_nans_cc = []
num_nans_dir = []
num_nans_xe = []
misclass_rate_cc = []
misclass_rate_xe = []
misclass_rate_dir = []
misclass_num_cc = []
misclass_num_xe = []
misclass_num_dir = []
for epoch in range(1, epochs + 1):
    iter = 0
    start = time.time()
    train_loglik_xe = []
    train_loglik_cc = []
    train_loglik_dir = []
    for train_x, train_y, train_hard in train_dataset:
        # We run gradient steps for all models
        obj = compute_apply_gradients(xe_model, train_x, train_y, train_hard, xe_optimizer)
        train_loglik_xe.append(obj)

        obj = compute_apply_gradients(cc_model, train_x, train_y, train_hard, optimizer)
        train_loglik_cc.append(obj)

        # Remove observations that cause numerical overflow for the Dirichlet
        temp_loglik = dir_model.compute_loglik(train_x, train_y)
        to_keep = tf.math.is_finite(temp_loglik)
        X_dir_train = train_x[to_keep]
        Y_dir_train = train_y[to_keep]
        hard_dir_train = train_hard[to_keep]
        obj = compute_apply_gradients(dir_model, X_dir_train, Y_dir_train, hard_dir_train, dir_optimizer)
        train_loglik_dir.append(obj)

    end = time.time()
    print("\n\nEpoch:", epoch, ". Time:", end - start)


    loglik_xe.append(sum(train_loglik_xe))
    print("XE loglik:", loglik_xe[-1])
    fitted = tf.math.softmax(xe_model(X_test))
    l2_xe_err.append(tf.reduce_mean(tf.sqrt(tf.losses.MSE(Y_test, fitted))))
    l1_xe_err.append(tf.reduce_mean(tf.losses.MAE(Y_test, fitted)))
    print('L2 XE test error:', l2_xe_err[-1])
    # Compute the number of test misclassifications:
    predictions = tf.math.argmax(fitted, axis=1)
    num_misclass = len(tf.where(test_labels != predictions))
    misclass_num_xe.append(num_misclass)
    misclass_rate_xe.append(num_misclass / 10000)
    print('XE # misclass:', num_misclass)

    loglik_cc.append(sum(train_loglik_cc))
    print("CC loglik:", loglik_cc[-1])
    temp_mean = cc_model.mean(X_test)
    to_keep = tf.logical_not(tf.reduce_any(tf.math.is_nan(temp_mean), axis=1))
    X_cc_test = X_test[to_keep]
    Y_cc_test = Y_test[to_keep]
    # Evaluate held-out likelihood
    fitted = temp_mean[to_keep]
    l2_cc_err.append(tf.reduce_mean(tf.sqrt(tf.losses.MSE(Y_cc_test, fitted))))
    l1_cc_err.append(tf.reduce_mean(tf.losses.MAE(Y_cc_test, fitted)))
    print('L2 CC test error:', l2_cc_err[-1])
    print('L1 CC test error:', l1_cc_err[-1])
    n = X_test.shape[0]
    aug_eta = tf.concat([cc_model(X_test), tf.zeros([n,1])], -1)
    predictions = tf.math.argmax(aug_eta, axis=1)
    num_misclass = len(tf.where(test_labels != predictions))
    misclass_rate_cc.append(num_misclass / 10000)
    misclass_num_cc.append(num_misclass)
    print('CC # misclass:', num_misclass)
    discard = tf.reduce_any(tf.math.is_nan(temp_mean), axis=1)
    num_nans_cc.append(len(tf.where(discard)))
    print("CC # nans:", num_nans_cc[-1])

    loglik_dir.append(sum(train_loglik_dir))
    print("Dir loglik:", loglik_dir[-1])
    fitted = dir_model.mean(X_test)
    l2_dir_err.append(tf.reduce_mean(tf.sqrt(tf.losses.MSE(Y_test, fitted))))
    l1_dir_err.append(tf.reduce_mean(tf.losses.MAE(Y_test, fitted)))
    print('L2 Dir test error:', l2_dir_err[-1])
    predictions = tf.math.argmax(dir_model(X_test), axis=1)
    num_misclass = len(tf.where(test_labels != predictions))
    print('Dir test misclassification rate:', num_misclass / 10000)
    misclass_rate_dir.append(num_misclass / 10000)
    misclass_num_dir.append(num_misclass)
    print('Dir # misclass:', num_misclass)
    temp_loglik = dir_model.compute_loglik(X_test, Y_test)
    discard = tf.logical_not(tf.math.is_finite(temp_loglik))
    num_nans_dir.append(len(tf.where(discard)))
    print("Dir # nans:", num_nans_dir[-1])

    if epoch % 10 == 0:
        results = pd.DataFrame({'epoch': np.arange(epoch),
                                'XE_L2_err': l2_xe_err,
                                'XE_L1_err': l2_xe_err,
                                'XE_misclass': misclass_num_xe,
                                'CC_L2_err': l2_cc_err,
                                'CC_L1_err': l1_cc_err,
                                'CC_misclass': misclass_num_cc,
                                'CC_nans': num_nans_cc,
                                'Dir_L2_err': l2_dir_err,
                                'Dir_L1_err': l1_dir_err,
                                'Dir_misclass': misclass_num_dir,
                                'Dir_nans': num_nans_dir
                                })

        # Save results:
        results.to_pickle(results_dir)

