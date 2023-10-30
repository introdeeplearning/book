import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plot_util

ax = plot_util.setup_axis((-3,3), (-.5,4))

x = np.linspace(-3, 3, 100)

mse_loss = tf.keras.losses.MeanSquaredError(
    reduction=tf.keras.losses.Reduction.NONE)
mae_loss = tf.keras.losses.MeanAbsoluteError(
    reduction=tf.keras.losses.Reduction.NONE)
huber_loss = tf.keras.losses.Huber(
    reduction=tf.keras.losses.Reduction.NONE)

zero = tf.zeros([100,1])

ax.plot(x, mse_loss(x.reshape([100,1]),zero)/2., 
        label='Scaled mean squared error')
ax.plot(x, mae_loss(x.reshape([100,1]),zero), 
        label='ℓ¹-error')
ax.plot(x, huber_loss(x.reshape([100,1]),zero), 
        label='1-Huber-error')
ax.legend()

plt.savefig("../../plots/huberloss.pdf", bbox_inches='tight')