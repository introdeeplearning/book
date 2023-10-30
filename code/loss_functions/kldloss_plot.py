import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plot_util

ax = plot_util.setup_axis((0,1), (0,3))

ax.set_aspect(.3)

x = np.linspace(0, 1, 100)

kld_loss = tf.keras.losses.KLDivergence(
    reduction=tf.keras.losses.Reduction.NONE)
cce_loss = tf.keras.losses.CategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE)
y = tf.constant([[0.3, 0.7]] * 100, shape=(100, 2))

X = tf.stack([x,1-x], axis=1)

ax.plot(x, kld_loss(y,X), label='Kullback-Leibler divergence')
ax.plot(x, cce_loss(y,X), label='Cross-entropy')
ax.legend()

plt.savefig("../../plots/kldloss.pdf", bbox_inches='tight')