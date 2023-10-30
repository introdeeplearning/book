import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(ax, g):
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    y = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    x, y = np.meshgrid(x, y)

    # flatten the grid to [num_points, 2] and convert to tensor
    grid = np.vstack([x.flatten(), y.flatten()]).T
    grid_torch = torch.from_numpy(grid).float()

    # pass the grid through the network
    z = g(grid_torch)

    # reshape the predictions back to a 2D grid
    Z = z.numpy().reshape(x.shape)

    # plot the heatmap
    ax.imshow(Z, origin='lower', extent=(-2 * np.pi, 2 * np.pi, 
                                         -2 * np.pi, 2 * np.pi))

M = 10000

def f(x):
    return torch.sin(x).prod(dim=1, keepdim=True)

torch.manual_seed(0)
X = torch.rand((M, 2)) * 4 * np.pi - 2 * np.pi
Y = f(X)

J = 32

N = 100000

loss = nn.MSELoss()
gamma = 0.05

fig, axs = plt.subplots(
    3, 3, figsize=(12, 12), sharex="col", sharey="row",
)

net = nn.Sequential(
    nn.Linear(2, 50), 
    nn.Softplus(), 
    nn.Linear(50,50), 
    nn.Softplus(), 
    nn.Linear(50, 1)
)

plot_after = [0, 100, 300, 1000, 3000, 10000, 30000, 100000]

for n in range(N + 1):
    indices = torch.randint(0, M, (J,))

    x = X[indices]
    y = Y[indices]

    net.zero_grad()

    loss_val = loss(net(x), y)
    loss_val.backward()

    with torch.no_grad():
        for p in net.parameters():
            p.sub_(gamma * p.grad)

    if n in plot_after:
        i = plot_after.index(n)

        with torch.no_grad():
            plot_heatmap(axs[i // 3][i % 3], net)
            axs[i // 3][i % 3].set_title(f"Batch {n}")

with torch.no_grad():
    plot_heatmap(axs[2][2], f)
    axs[2][2].set_title("Target")

plt.tight_layout()
plt.savefig("../../plots/sgd2.pdf", bbox_inches="tight")
