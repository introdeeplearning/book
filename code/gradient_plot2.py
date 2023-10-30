import numpy as np
import matplotlib.pyplot as plt

K = [1., 10.]
vartheta = np.array([1., 1.])

def f(x, y):
    result =  K[0] / 2. * np.abs(x - vartheta[0])**2 \
    + K[1] / 2. * np.abs(y - vartheta[1])**2 
    return result

def nabla_f(x):
    return K * (x - vartheta)

plt.figure()

# Plot contour lines of f
x = np.linspace(-3., 7., 100)
y = np.linspace(-2., 4., 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
cp = plt.contour(X, Y, Z, colors="black", 
                 levels = [0.5,2,4,8,16],
                 linestyles=":")

# Plot arrows along contour lines
for l in [0.5,2,4,8,16]:
  for d in np.linspace(0, 2.*np.pi, 10, endpoint=False):
    x = np.cos(d) / ((K[0] / (2*l))**.5) + vartheta[0]
    y = np.sin(d) / ((K[1] / (2*l))**.5) + vartheta[1]
    grad = nabla_f(np.array([x,y]))
    plt.arrow(x, y, -.05 * grad[0], -.05 * grad[1],
      length_includes_head=True, head_width=.08, 
      head_length=.1, color='b')

plt.savefig("../plots/gradient_plot2.pdf")