import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return x**4 - 3 * x**2

def nabla_f(x):
  return 4 * x**3 - 6 * x

plt.figure()

# Plot graph of f
x = np.linspace(-2,2,100)
plt.plot(x,f(x))

# Plot arrows
for x in np.linspace(-1.9,1.9,21):
  d = nabla_f(x)
  plt.arrow(x, f(x), -.05 * d, 0,
    length_includes_head=True, head_width=0.08,
    head_length=0.05, color='b')

plt.savefig("../plots/gradient_plot1.pdf")