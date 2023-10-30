import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plot_util

ax = plot_util.setup_axis((-4,3), (-.5,3))

x = np.linspace(-4, 3, 100)

ax.plot(x, tf.keras.activations.relu(x), label='ReLU')
ax.plot(x, tf.keras.activations.softplus(x), label='softplus')
ax.plot(x, tf.keras.activations.gelu(x), label='GELU')
ax.legend()

plt.savefig("../../plots/gelu.pdf", bbox_inches='tight')