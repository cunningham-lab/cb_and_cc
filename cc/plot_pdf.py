# Makes heatmaps of CC pdfs

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ternary
from cc_funcs import lambda_to_eta, cc_log_normalizer
import tensorflow as tf
fig = plt.figure()
matplotlib.rcParams['text.usetex'] = True
print(plt.rcParams["figure.figsize"])
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 5.0  # Landscape A4 format inches
fig_size[1] = 4.8
# plt.rcParams["figure.figsize"] = fig_size
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 30}
matplotlib.rc('font', **font)
plt.rcParams.update({'axes.titlesize': 'medium'})

# l = np.array([.13,.131,.129])
# l_text = '[0.33, 0.33, 0.33]'

l = np.array([0.491,0.49,0.019])
l_text = '[0.49, 0.49, 0.02]'

# l = np.array([0.7, 0.2, 0.1])
# l_text = '[0.7, 0.2, 0.1]'

# l = np.array([0.8,0.11,0.09])
# l_text = '[0.8, 0.1, 0.1]'

l = l / l.sum()

def f(x):
    # Compute the pdf for a 2-dimensional (K=3) CC variate
    lam = tf.convert_to_tensor(l.reshape([1,-1]), dtype='float32')
    eta = lambda_to_eta(lam)
    res = np.exp(eta[0,0] * x[0] + eta[0,1] * x[1] - cc_log_normalizer(eta))
    return res[0]



figure, tax = ternary.figure(scale=60)
# Toggle colorbar true/false here
tax.heatmapf(f, boundary=True, style="triangular", vmin=0, vmax=6, colorbar=False, scientific=False, cbarlabel="density")
tax.boundary(linewidth=2.0)
tax.set_title("$\lambda$ = {}".format(l_text), fontweight='heavy')
tax.get_axes().axis('off')
print(tax.get_axes())
tax.bottom_axis_label('$x_1$')
tax.left_axis_label('$x_2$')
tax.right_axis_label('$x_3$')
tax.show()
figure.savefig('density_plot.pdf')

