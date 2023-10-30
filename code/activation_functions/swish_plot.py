import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plot_util

ax = plot_util.setup_axis((-4,3), (-.5,3))

x = np.linspace(-4, 3, 100)

ax.plot(x, tf.keras.activations.relu(x), label='ReLU')
ax.plot(x, tf.keras.activations.gelu(x), label='GELU')
ax.plot(x, tf.keras.activations.swish(x), label='swish')
ax.legend()

plt.savefig("../../plots/swish.pdf", bbox_inches='tight')