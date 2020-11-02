"""Creates plots of the learned representations from our models.

Requires that all our label-smoothing runs already be in place.
Plots the penultimate layer activations under our different models,
for training and validation samples, with/without LS and CC-LS.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from cifar10_model import LS_model
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Set up the plotting parameters
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
font = {'weight': 'normal',
        'size': 18}
matplotlib.rc('font', **font)
colors = ['blue', 'orange', 'pink']
subplots = [231, 232, 233, 234, 235, 236]
plt.figure(figsize=(15, 9))

# Set up the saved parameters
optimizer = 'Adam'
lr = 0.001
wd = 0.0
alpha = 0.1
seed = 0
notes = ''
losses = ['XE', 'LS', 'CC', 'XE', 'LS', 'CC']

# Load the data and normalize
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
model = LS_model(train=False)
x_train, x_test = model.normalize(x_train, x_test)

def make_basis(model):
    """Find the 2 basis vectors for the plane spanning the 3 template vectors

    First extracts the 3 'template' vectors from our model. These vectors
    correspond to the weights associated to 3 (arbitrarily chosen) classes
    in our penultimate layer. We then find an orthonormal basis for the
    plane that goes through these 3 points.
    """
    v0 = tf.concat([model.weights[-2][:,0], tf.reshape(model.weights[-1][0], [1])], axis=0)
    v1 = tf.concat([model.weights[-2][:,1], tf.reshape(model.weights[-1][1], [1])], axis=0) - v0
    v2 = tf.concat([model.weights[-2][:,2], tf.reshape(model.weights[-1][2], [1])], axis=0) - v0
    e1 = v1 / tf.sqrt(tf.reduce_sum(tf.square(v1)))
    e2 = (v2 - tf.reduce_sum(e1 * v2) * v2)
    e2 = e2 / tf.sqrt(tf.reduce_sum(tf.square(e2)))
    return e1, e2

# Now loop through our 3 models (both for training and validation samples)
for i in range(len(losses)):
    loss = losses[i]
    plt.subplot(subplots[i])
    # Add labels to the plots
    if subplots[i] == 231:
        plt.ylabel('Training samples')
    if subplots[i] == 234:
        plt.ylabel('Test samples')
    if i == 0:
        plt.title('w/o LS')
    if i == 1:
        plt.title('with LS')
    if i == 2:
        plt.title('CC-LS')

    # Load the appropriate model
    results_path = "./tf_models/cifar10_CNN_loss{}_opt{}_lr{}_wd{}_alpha{}_seed{}{}.h5".format(loss, optimizer, lr, wd, alpha, seed, notes)
    model = tf.keras.models.load_model(results_path)

    # Construct a basis:
    e1, e2 = make_basis(model)

    # Use the basis to calculate the projections
    xs = []
    ys = []
    within_vars = []
    for categ in [0, 1, 2]:
        if i < len(losses) / 2: # top row of our plot is for training samples
            samples = x_train[tf.reshape(y_train, [-1]) == categ][0:100]
        else: # bottom row is for testing samples
            samples = x_test[tf.reshape(y_test, [-1]) == categ][0:100]

        # Compute the projections by running our samples through to the
        # penultimate layer, and dot producting with basis
        activations = samples
        for l in range(len(model.layers) - 1):
            activations = model.layers[l](activations)

        # Add intercept for the biases
        paddings = tf.constant([[0, 0], [0, 1]])
        padded_activations = tf.pad(activations, paddings, constant_values=1.0)
        y = tf.reduce_sum(padded_activations * e1, axis=1).numpy()
        x = tf.reduce_sum(padded_activations * e2, axis=1).numpy()
        plt.scatter(x, y, color=colors[categ])

# Save the figure and show
plt.savefig('./../out/ls_rep.pdf')
plt.show()

# Now we go over the same loop again, but instead of plotting we compute
# the proportion of variance explained (WCSS / BCSS). We repeat over 10
# seeds in order to derive 'error bars' for the WCSS / BCSS metric
table = np.zeros([10, 6])
for seed in range(10):
    for i in range(len(losses)):
        loss = losses[i]
        results_path = "./tf_models/cifar10_CNN_loss{}_opt{}_lr{}_wd{}_alpha{}_seed{}{}.h5".format(loss, optimizer, lr, wd, alpha, seed, notes)
        model = tf.keras.models.load_model(results_path)
        e1, e2 = make_basis(model)

        xs = []
        ys = []
        within_vars = []
        for categ in [0, 1, 2]:
            if i < len(losses) / 2:
                samples = x_train[tf.reshape(y_train, [-1]) == categ][0:100]
            else:
                samples = x_test[tf.reshape(y_test, [-1]) == categ][0:100]

            activations = samples
            for l in range(len(model.layers) - 1):
                activations = model.layers[l](activations)

            paddings = tf.constant([[0, 0], [0, 1]])
            padded_activations = tf.pad(activations, paddings, constant_values=1.0)
            y = tf.reduce_sum(padded_activations * e1, axis=1).numpy()
            x = tf.reduce_sum(padded_activations * e2, axis=1).numpy()
            ys.append(y)
            xs.append(x)
            D = np.concatenate([x.reshape([-1,1]),y.reshape([-1,1])], axis=1)
            within_vars.append(np.mean(np.var(D, axis=0)))

        within = np.sqrt(np.mean(within_vars))

        x = np.concatenate(xs)
        y = np.concatenate(ys)
        D = np.concatenate([x.reshape([-1, 1]), y.reshape([-1, 1])], axis=1)
        between = np.sqrt(np.mean(np.var(D, axis=0)))

        print("Ratio")
        print(within / between)

        table[seed, i] = np.square(within / between)

# Output results
print(table)
print(np.mean(table, axis=0))
print(np.std(table, axis=0))