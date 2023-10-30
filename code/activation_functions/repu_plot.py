import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plot_util

ax = plot_util.setup_axis((-2,2), (-.5,3))
ax.set_ylim(-.5, 3)

x = np.linspace(-2, 2, 100)

ax.plot(x, tf.keras.activations.relu(x), linewidth=3, label='ReLU')
ax.plot(x, tf.keras.activations.relu(x)**2, label='RePU')
ax.legend()

plt.savefig("../../plots/repu.pdf", bbox_inches='tight')