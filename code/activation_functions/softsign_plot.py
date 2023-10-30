import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plot_util

ax = plot_util.setup_axis((-5,5), (-1.5,1.5))

x = np.linspace(-5, 5, 100)

ax.plot(x, tf.keras.activations.tanh(x), label='tanh')
ax.plot(x, tf.keras.activations.softsign(x), label='softsign')
ax.legend()

plt.savefig("../../plots/softsign.pdf", bbox_inches='tight')