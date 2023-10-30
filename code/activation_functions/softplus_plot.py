import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plot_util

ax = plot_util.setup_axis((-4,4), (-.5,4))

x = np.linspace(-4, 4, 100)

ax.plot(x, tf.keras.activations.relu(x), label='ReLU')
ax.plot(x, tf.keras.activations.softplus(x), label='softplus')
ax.legend()

plt.savefig("../../plots/softplus.pdf", bbox_inches='tight')