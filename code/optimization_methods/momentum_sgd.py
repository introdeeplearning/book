import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

M = 10000

torch.manual_seed(0)
X = torch.rand((M, 1)) * 4 * np.pi - 2 * np.pi
Y = torch.sin(X)

J = 64

N = 100000

loss = nn.MSELoss()
lr = 0.01
alpha = 0.999

fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey='row')

net = nn.Sequential(
    nn.Linear(1, 200), nn.ReLU(), nn.Linear(200, 1)
)

for i, alpha in enumerate([0, 0.9, 0.99, 0.999]):
    print(f"alpha = {alpha}")

    for lr in [0.1, 0.03, 0.01, 0.003]:
        torch.manual_seed(0)
        net.apply(
            lambda m: m.reset_parameters()
            if isinstance(m, nn.Linear)
            else None
        )

        momentum = [
            p.clone().detach().zero_() for p in net.parameters()
        ]

        losses = []
        print(f"lr = {lr}")

        for n in range(N):
            indices = torch.randint(0, M, (J,))

            x = X[indices]
            y = Y[indices]

            net.zero_grad()

            loss_val = loss(net(x), y)
            loss_val.backward()

            with torch.no_grad():
                for m, p in zip(momentum, net.parameters()):
                    m.mul_(alpha)
                    m.add_((1 - alpha) * p.grad)
                    p.sub_(lr * m)

            if n % 100 == 0:
                with torch.no_grad():
                    x = (torch.rand((1000, 1)) - 0.5) * 4 * np.pi
                    y = torch.sin(x)
                    loss_val = loss(net(x), y)
                    losses.append(loss_val.item())

        axs[i].plot(losses, label=f"$\\gamma = {lr}$")

    axs[i].set_yscale("log")
    axs[i].set_ylim([1e-6, 1])
    axs[i].set_title(f"$\\alpha = {alpha}$")

axs[0].legend()

plt.tight_layout()
plt.savefig("../plots/sgd_momentum.pdf", bbox_inches='tight')
