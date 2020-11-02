"""Fits a CNN on CIFAR-10 with different regularization types.

To be called multiple times under different parameter settings, to
train a set of CNNs on CIFAR-10 under combinations of dropout,
weight decay, batch normalization, label smoothing and CC-LS.
"""

from __future__ import print_function
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
import numpy as np
from tensorflow.keras import regularizers
import pandas as pd
import time
import argparse
import sys
sys.path.append('../cc')
from cc_funcs import cc_log_prob, cc_mean, cc_log_norm_const

# ------------------------------------------
# Set up parameters
# ------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('--epochs', dest='epochs', type=int, default=500)
parser.add_argument('--lr', dest='lr', type=float, default=0.001)
parser.add_argument('--wd', dest='wd', type=float, default=0.0)
parser.add_argument('--alpha', dest='alpha', type=float, default=0.1)
parser.add_argument('--dp', dest='dp', type=str, default="Yes")
parser.add_argument('--bn', dest='bn', type=str, default="Yes")
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)
parser.add_argument('--notes', dest='notes', type=str, default='')
parser.add_argument('--opt', dest='opt', type=str, default='Adam')
parser.add_argument('--loss', dest='loss', type=str, default='CC')
parser.add_argument('--save', dest='save', type=str, default="No")

args = parser.parse_args()

seed = args.seed
epochs = args.epochs
lr = args.lr
wd = args.wd
alpha = args.alpha
dp = args.dp
bn = args.bn
loss = args.loss
optimizer = args.opt
batch_size = args.batch_size
save_weights = args.save
notes = args.notes

# Fixed parameters
wd_conv = 0.0 # weight decay used for convolutional layers
num_classes = 10 # number of output classes

# Set seed
np.random.seed(seed)
tf.random.set_seed(seed)

# Set up our save file path
if dp == "No":
    notes = notes + 'noDP'
if bn == "No":
    notes = notes + 'noBN'
file_name = "cifar10_CNN_loss{}_opt{}_lr{}_wd{}_alpha{}_seed{}{}".format(loss, optimizer, lr, wd, alpha, seed, notes)
print(file_name)
results_path = "./out/" + file_name + ".csv"

# Set up the optimizer
if optimizer == 'SGD':
    # lr_decay = 1e-6
    lr_drop = 20
    def lr_scheduler(epoch):
        return lr * (0.5 ** (epoch // lr_drop))
    reduce_lr = tensorflow.keras.callbacks.LearningRateScheduler(lr_scheduler)
    opt = optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
elif optimizer == 'RMSprop':
    def lr_scheduler(epoch):
        lr = 0.001
        if epoch > 75:
            lr = 0.0005
        if epoch > 100:
            lr = 0.0003
        return lr
    opt = optimizers.RMSprop(lr=0.001, decay=1e-6)
elif optimizer == 'Adam':
    opt = optimizers.Adam(lr=lr)
else:
    raise ValueError("Optimizer not implemented")

# Set up the loss function
def loss_func_CC(yLS, y_pred):
    """Declare CC loss function out here as it will be shared between CC and RCC
    
    Args:
        yLS, y_pred: Tensors of observed values and fitted values. The fitted
        values y_pred are in the natural parameter (eta) space, with a
        redundant Kth component (which we remove prior to calling the CC pdf).

    Returns:
        A scalar, the negative log likelihood of the sample
    """
    # Remove the redundant component from eta
    eta = y_pred[:, 0:-1] - tf.reshape(y_pred[:, -1], [-1, 1])
    # Add our logic to deal with CC numerical instabilities
    temp_mean = tf.stop_gradient(cc_mean(eta))
    discard = tf.math.reduce_any(tf.logical_or(temp_mean < 0, tf.math.is_nan(temp_mean)), axis=1)
    keep = tf.logical_not(discard)
    logits = tf.pad(eta, [[0, 0], [0, 1]])

    # Keep the XE term only for the numerically unstable gradients
    partial = - tf.nn.softmax_cross_entropy_with_logits(yLS[discard], logits[discard])
    full = cc_log_prob(yLS[keep], eta[keep])
    return - tf.reduce_sum(partial) - tf.reduce_sum(full)

if loss == "XE":
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
elif loss == "LS":
    def loss_func(y_true, y_pred):
        """Returns the label-smoothed XE loss"""
        yLS = y_true * (1 - alpha) + alpha / num_classes
        return tf.losses.categorical_crossentropy(yLS, y_pred, from_logits=True)
elif loss == "RLS":
    def loss_func(y_true, y_pred):
        """Returns the 'continuously randomized' label smoothing loss"""
        n, K = y_true.shape
        # Instead of adding a fixed vector to all labels, we add noise
        u = np.random.dirichlet(alpha=np.ones(K), size=n)
        yLS = y_true * (1 - alpha) + alpha * u
        return tf.losses.categorical_crossentropy(yLS, y_pred, from_logits=True)
elif loss == "MSE":
    def loss_func(y_true, y_pred):
        """Returns the MSE loss (unused)"""
        return tf.square(y_true - y_pred)
elif loss == "CC":
    def loss_func(y_true, y_pred):
        """Returns the CC-LS loss"""
        yLS = y_true * (1 - alpha) + alpha / num_classes
        return loss_func_CC(yLS, y_pred)
elif loss == "RCC":
    def loss_func(y_true, y_pred):
        """Returns the 'continuously randomized' CC-LS loss"""
        n, K = y_true.shape
        u = np.random.dirichlet(alpha=np.ones(K), size=n)
        yLS = y_true * (1 - alpha) + alpha * u
        return loss_func_CC(yLS, y_pred)

class LS_model:
    def __init__(self, train=True):
        self.num_classes = num_classes
        self.weight_decay = 0.0005
        self.x_shape = [32, 32, 3]
        self.history = None
        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)

    def build_model(self):
        """Creates our Keras model object"""

        model = Sequential()

        # Architecture adapted from: https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
        model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wd_conv)))
        model.add(Activation('elu'))
        if bn == "Yes":
            model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wd_conv)))
        model.add(Activation('elu'))
        if bn == "Yes":
            model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if dp == "Yes":
            model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wd_conv)))
        model.add(Activation('elu'))
        if bn == "Yes":
            model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wd_conv)))
        model.add(Activation('elu'))
        if bn == "Yes":
            model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if dp == "Yes":
            model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wd_conv)))
        model.add(Activation('elu'))
        if bn == "Yes":
            model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wd_conv)))
        model.add(Activation('elu'))
        if bn == "Yes":
            model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if dp == "Yes":
            model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(self.num_classes, kernel_regularizer=regularizers.l2(wd)))

        return model

    def normalize(self, X_train, X_test):
        """Standardizes the data prior to training/validation"""
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean)/(std + 1e-7)
        X_test = (X_test - mean)/(std + 1e-7)
        return X_train, X_test

    def train(self, model):
        """Trains our model and stores validation errors at each epoch"""

        # Load the data and standardize:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)
        y_train = tensorflow.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tensorflow.keras.utils.to_categorical(y_test, self.num_classes)

        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # compile model
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

        # custom training loop:
        res = []
        for epoch in range(0, epochs + 1):
            start = time.time() # Time our epochs
            print('epoch', epoch)

            # Compute the validation score and output
            val_res = model.evaluate(x_test, y_test, verbose=0)
            res.append(val_res)
            print(val_res)

            minibatch_iter = 0
            for train_x, train_y in datagen.flow(x_train, y_train, batch_size=batch_size):
                with tf.GradientTape() as tape:
                    logits = model(train_x)
                    negloglik = loss_func(train_y, logits)
                gradients = tape.gradient(negloglik, model.trainable_variables)
                if all([tf.reduce_all(tf.math.is_finite(g)) for g in gradients]):
                    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                else:
                    print("Have NaNs in gradients!")
                    print(minibatch_iter)
                minibatch_iter += 1

                if minibatch_iter > 50000 / batch_size:
                    break

            # Update learning rates (for SGD and RMSprop):
            if optimizer in ['SGD', 'RMSprop']:
                model.optimizer.lr = lr_scheduler(epoch)

            end = time.time()
            print(end - start)
            print('')

            # Every 50 epochs we save results
            if epoch % 50 == 49 or epoch == epochs:
                results = pd.DataFrame(res, columns=['val_XE', 'val_accuracy'])
                results.to_csv(results_path)
                if save_weights == 'Yes':
                    save_path = "./tf_models/" + file_name + ".h5"
                    model.save(save_path)

        return model

if __name__ == '__main__':
    model = LS_model()
