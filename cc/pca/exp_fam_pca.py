# Linear dimensionality reduction of election data

import tensorflow as tf
import tensorflow_probability as tfp
from cc_funcs import cc_log_prob, cc_mean, eta_to_lambda
from election_data import load_election_data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

seed = 0
epochs = 500
lr_cc = 0.1
lr_dir = 0.1

dtype = 'float32'

tf.random.set_seed(seed)
tfd = tfp.distributions
num_parties = 4
df = load_election_data(num_parties)

class CCPCA(object):
    def __init__(self, zdim, xdim, n):
        self.n = n
        self.zdim = zdim
        self.xdim = xdim
        self.Z = tf.Variable(tf.random.normal([self.n, self.zdim], dtype=dtype))
        self.W_dec = tf.Variable(tf.random.normal([self.zdim, self.xdim - 1], dtype=dtype))
        self.b_dec = tf.Variable(tf.random.normal([self.xdim - 1], dtype=dtype))

    def decode(self, Z):
        eta = self.b_dec + tf.linalg.matmul(Z, self.W_dec)
        return eta

    def obj_func(self, X):
        eta = self.b_dec + tf.linalg.matmul(self.Z, self.W_dec)
        temp = tf.stop_gradient(cc_log_prob(X, eta))
        to_keep = tf.math.is_finite(temp)
        reconstruction_loss = cc_log_prob(X[to_keep], eta[to_keep])
        return reconstruction_loss


class DPCA(object):
    def __init__(self, zdim, xdim, n):
        self.n = n
        self.zdim = zdim
        self.xdim = xdim
        self.Z = tf.Variable(tf.random.normal([self.n, self.zdim], dtype=dtype))
        self.W_dec = tf.Variable(tf.random.normal([self.zdim, self.xdim], dtype=dtype))
        self.b_dec = tf.Variable(tf.random.normal([self.xdim], dtype=dtype))

    def decode(self, Z):
        alpha = tf.math.exp(self.b_dec + tf.linalg.matmul(Z, self.W_dec))
        return alpha

    def obj_func(self, X):
        alpha = tf.math.exp(self.b_dec + tf.linalg.matmul(self.Z, self.W_dec))
        dist = tfd.Dirichlet(alpha)
        temp = tf.stop_gradient(dist.log_prob(X))
        to_keep = tf.math.is_finite(temp)
        dist = tfd.Dirichlet(alpha[to_keep])
        reconstruction_loss = dist.log_prob(X[to_keep])
        return reconstruction_loss


test_idx = (df['test'] == 1).values
Y_train = df.iloc[~test_idx, 0:num_parties+1].values
Y_test = df.iloc[test_idx, 0:num_parties+1].values
# Move Y away from zero
Y_train = Y_train + 1e-3 / (1 + 1e-3 * (num_parties))
Y_train = tf.convert_to_tensor(Y_train, dtype=dtype)
Y_test = Y_test + 1e-3 / (1 + 1e-3 * (num_parties))
Y_test = tf.convert_to_tensor(Y_test, dtype=dtype)
X_train = df.iloc[~test_idx, num_parties+1:-1].values
X_train = tf.convert_to_tensor(X_train, dtype=dtype)
X_test = df.iloc[test_idx, num_parties+1:-1].values
X_test = tf.convert_to_tensor(X_test, dtype=dtype)
num_predictors = X_test.shape[1]

cc_model = CCPCA(2, num_parties + 1, X_train.shape[0])
dir_model = DPCA(2, num_parties + 1, X_train.shape[0])
cc_optimizer = tf.keras.optimizers.Adam(lr_cc)
dir_optimizer = tf.keras.optimizers.Adam(lr_dir)

def compute_apply_gradients(model, X, optimizer):
    with tf.GradientTape() as tape:
        # prior = - tf.reduce_sum(tf.square(model.W)) - tf.reduce_sum(tf.square(model.b))
        negloglik = tf.reduce_sum(-model.obj_func(X))
    # dW_enc, dW_dec, db_enc, db_dec = tape.gradient(negloglik, [model.W_enc, model.W_dec, model.b_enc, model.b_dec])
    dZ, dW_dec, db_dec = tape.gradient(negloglik, [model.Z, model.W_dec, model.b_dec])
    optimizer.apply_gradients(zip([dZ, dW_dec, db_dec], [model.Z, model.W_dec, model.b_dec]))
    return - negloglik

elbo_cc = []
elbo_dir = []
l2_cc_err = []
l1_cc_err = []
l2_dir_err = []
l1_dir_err = []
for epoch in range(1, epochs + 1):
    print('epoch:', epoch)
    obj = compute_apply_gradients(cc_model, Y_train, cc_optimizer)
    elbo_cc.append(obj)
    print('CC obj_func:', elbo_cc[-1])

    obj = compute_apply_gradients(dir_model, Y_train, dir_optimizer)
    elbo_dir.append(obj)
    print('Dir obj_func:', elbo_dir[-1])


# Plot latent representations

custom_lines2 = [Line2D([0], [0], lw=3, color='brown'),
                Line2D([0], [0], lw=3, color='green')]

ind1 = tf.cast(X_train[:,3], dtype='bool') # N. Ireland
ind2 = tf.cast(X_train[:,4], dtype='bool') # Scotland
ind3 = tf.cast(X_train[:,5], dtype='bool') # Wales
ind0 = tf.logical_not(tf.logical_or(ind1, tf.logical_or(ind2, ind3)))


###################################
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
###################################

###################################
# Storing colors here
blue = "#377eb8"
purple = "#984ea3"
orange = "#ff7f00"
brown = "#a65628"
pink = "#f781bf"
grey = "#999999"
####################################

matplotlib.rc('xtick', labelsize=13)
matplotlib.rc('ytick', labelsize=13)
font = {'weight' : 'normal',
        'size'   : 13}
matplotlib.rc('font', **font)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

latents = cc_model.Z
latents = StandardScaler().fit_transform(latents.numpy())

ax1.scatter(latents[ind0][:,0], latents[ind0][:,1], c = 'black', label='England', marker='x')
ax1.scatter(latents[ind1][:,0], latents[ind1][:,1], c='green', label='N. Ireland', marker='*')
ax1.scatter(latents[ind2][:,0], latents[ind2][:,1], c='blue', label='Scotland', marker='+')
ax1.scatter(latents[ind3][:,0], latents[ind3][:,1], c='red', label='Wales', marker='o')
# ax1.legend(loc='upper left')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title("CC-PCA")
# plt.show()


dir_latents = dir_model.Z
dir_latents = StandardScaler().fit_transform(dir_latents.numpy())

ax2.scatter(dir_latents[ind0][:,0], dir_latents[ind0][:,1], c = 'black', label='England', marker='x')
ax2.scatter(dir_latents[ind1][:,0], dir_latents[ind1][:,1], c='green', label='N. Ireland', marker='*')
ax2.scatter(dir_latents[ind2][:,0], dir_latents[ind2][:,1], c='blue', label='Scotland', marker='+')
ax2.scatter(dir_latents[ind3][:,0], dir_latents[ind3][:,1], c='red', label='Wales', marker='o')
ax2.legend(loc='lower left')
ax2.set_xlabel('PC1')
# ax2.set_ylabel('PC2')
ax2.set_title("Dirichlet-PCA")
# ax2.show()

Y_clr = np.log(Y_train) - np.mean(np.log(Y_train.numpy()), axis=1, keepdims=True)
x = StandardScaler().fit_transform(Y_clr)
latents = PCA(n_components=2).fit_transform(x)

ax3.scatter(latents[ind1][:,0], latents[ind1][:,1], c='green', label='N. Ireland', marker='*')
ax3.scatter(latents[ind0][:,0], latents[ind0][:,1], c = 'black', label='England', marker='x')
ax3.scatter(latents[ind2][:,0], latents[ind2][:,1], c='blue', label='Scotland', marker='+')
ax3.scatter(latents[ind3][:,0], latents[ind3][:,1], c='red', label='Wales', marker='o')
# ax3.legend()
ax3.set_xlabel('PC1')
# ax3.set_ylabel('PC2')
ax3.set_title("clr-PCA")

fig.tight_layout()
fig.savefig('../election_dr.pdf')
plt.show()

