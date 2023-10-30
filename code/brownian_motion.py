import numpy as np
import matplotlib.pyplot as plt

def generate_brownian_motion(T, N):
    increments = np.random.randn(N) * np.sqrt(T/N)
    BM = np.cumsum(increments)
    BM = np.insert(BM, 0, 0)
    return BM

T = 1
N = 1000
t_values = np.linspace(0, T, N+1)

fig, axarr = plt.subplots(2, 2)

for i in range(2):
    for j in range(2):
        BM = generate_brownian_motion(T, N)
        axarr[i, j].plot(t_values, BM)

plt.tight_layout()
plt.savefig('../plots/brownian_motions.pdf')
plt.show()