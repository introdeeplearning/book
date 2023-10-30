import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plot_util

ax = plot_util.setup_axis((-3,3), (-.5,1.5))

x = np.linspace(-3, 3, 100)

ax.plot(x, tf.keras.activations.relu(x, max_value=1), 
        label='(0,1)-clipping')
ax.plot(x, tf.keras.activations.sigmoid(x), 
        label='standard logistic')
ax.legend()

plt.savefig("../../plots/logistic.pdf", bbox_inches='tight')