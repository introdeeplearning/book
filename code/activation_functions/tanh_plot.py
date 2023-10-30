import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plot_util

ax = plot_util.setup_axis((-3,3), (-1.5,1.5))

x = np.linspace(-3, 3, 100)

ax.plot(x, tf.keras.activations.relu(x+1, max_value=2)-1, 
        label='(-1,1)-clipping')
ax.plot(x, tf.keras.activations.sigmoid(x), 
        label='standard logistic')
ax.plot(x, tf.keras.activations.tanh(x), label='tanh')
ax.legend()

plt.savefig("../../plots/tanh.pdf", bbox_inches='tight')