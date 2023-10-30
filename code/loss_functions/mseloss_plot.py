import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plot_util

ax = plot_util.setup_axis((-2,2), (-.5,2))

x = np.linspace(-2, 2, 100)

mse_loss = tf.keras.losses.MeanSquaredError(
    reduction=tf.keras.losses.Reduction.NONE)
zero = tf.zeros([100,1])

ax.plot(x, mse_loss(x.reshape([100,1]),zero), 
        label='Mean squared error')
ax.legend()

plt.savefig("../../plots/mseloss.pdf", bbox_inches='tight')