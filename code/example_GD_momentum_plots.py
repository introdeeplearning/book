# Example for GD and momentum GD

import numpy as np
import matplotlib.pyplot as plt

# Number of steps for the schemes
N = 8

# Problem setting
d = 2
K = [1., 10.]

vartheta = np.array([1., 1.])
xi = np.array([5., 3.])

def f(x, y):
    result =  K[0] / 2. * np.abs(x - vartheta[0])**2 \
    + K[1] / 2. * np.abs(y - vartheta[1])**2 
    return result

def nabla_f(x):
    return K * (x - vartheta)

# Coefficients for GD
gamma_GD = 2 /(K[0] + K[1])

# Coefficients for momentum
gamma_momentum = 0.3
alpha = 0.5 

# Placeholder for processes
Theta = np.zeros((N+1, d))
M = np.zeros((N+1, d))
m = np.zeros((N+1, d))

Theta[0] = xi
M[0] = xi

# Perform gradient descent 
for i in range(N):
    Theta[i+1] = Theta[i] - gamma_GD * nabla_f(Theta[i])

# Perform momentum GD 
for i in range(N):
    m[i+1] = alpha * m[i] + (1 - alpha) * nabla_f(M[i])
    M[i+1] = M[i] - gamma_momentum * m[i+1]


### Plot ###
plt.figure()

# Plot the gradient descent process
plt.plot(Theta[:, 0], Theta[:, 1], 
         label = "GD", color = "c", 
         linestyle = "--", marker = "*")

# Plot the momentum gradient descent process
plt.plot(M[:, 0], M[:, 1], 
         label = "Momentum", color = "orange", marker = "*")

# Target value
plt.scatter(vartheta[0],vartheta[1], 
            label = "vartheta", color = "red", marker = "x")

# Plot contour lines of f
x = np.linspace(-3., 7., 100)
y = np.linspace(-2., 4., 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
cp = plt.contour(X, Y, Z, colors="black", 
                 levels = [0.5,2,4,8,16],
                 linestyles=":")

plt.legend()
plt.savefig("../plots/GD_momentum_plots.pdf")
