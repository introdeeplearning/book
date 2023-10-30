import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plot_util

ax = plot_util.setup_axis((-2,2), (-.5,2))

x = np.linspace(-2, 2, 100)

ax.plot(x, tf.keras.activations.relu(x), linewidth=3, label='ReLU')
ax.plot(x, tf.keras.activations.relu(x, alpha=0.1), 
        label='leaky ReLU')
ax.legend()

plt.savefig("../../plots/leaky_relu.pdf", bbox_inches='tight')