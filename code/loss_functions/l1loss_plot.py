import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plot_util

ax = plot_util.setup_axis((-2,2), (-.5,2))

x = np.linspace(-2, 2, 100)

mae_loss = tf.keras.losses.MeanAbsoluteError(
    reduction=tf.keras.losses.Reduction.NONE)
zero = tf.zeros([100,1])

ax.plot(x, mae_loss(x.reshape([100,1]),zero), 
        label='ℓ¹-error')
ax.legend()

plt.savefig("../../plots/l1loss.pdf", bbox_inches='tight')