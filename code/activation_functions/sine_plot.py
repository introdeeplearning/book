import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plot_util

ax = plot_util.setup_axis((-2*np.pi,2*np.pi), (-1.5,1.5))

x = np.linspace(-2*np.pi, 2*np.pi, 100)

ax.plot(x, np.sin(x))

plt.savefig("../../plots/sine.pdf", bbox_inches='tight')