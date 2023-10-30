import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plot_util

ax = plot_util.setup_axis((-3,3), (-.5,1.5))

x = np.linspace(-3, 3, 100)

ax.plot(x[0:50], [0]*50, 'C0')
ax.plot(x[50:100], [1]*50, 'C0', label='Heaviside')
ax.plot(x, tf.keras.activations.sigmoid(x), 'C1', 
        label='standard logistic')
ax.legend()

plt.savefig("../../plots/heaviside.pdf", bbox_inches='tight')