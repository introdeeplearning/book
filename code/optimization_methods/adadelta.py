import torch
import torch.nn as nn
import numpy as np

net = nn.Sequential(
    nn.Linear(1, 200), nn.ReLU(), nn.Linear(200, 1)
)

M = 1000

X = torch.rand((M, 1)) * 4 * np.pi - 2 * np.pi
Y = torch.sin(X)

J = 64

N = 150000

loss = nn.MSELoss()
beta = 0.9
delta = 0.9
eps = 1e-10

moments = [p.clone().detach().zero_() for p in net.parameters()]
Delta = [p.clone().detach().zero_() for p in net.parameters()]

for n in range(N):
    indices = torch.randint(0, M, (J,))

    x = X[indices]
    y = Y[indices]

    net.zero_grad()

    loss_val = loss(net(x), y)
    loss_val.backward()

    with torch.no_grad():
        for m, D, p in zip(moments, Delta, net.parameters()):
            m.mul_(beta)
            m.add_((1 - beta) * p.grad * p.grad)
            inc = ((eps + D) / (eps + m)).sqrt() * p.grad
            p.sub_(inc)
            D.mul_(delta)
            D.add_((1 - delta) * inc * inc)

    if n % 1000 == 0:
        with torch.no_grad():
            x = torch.rand((1000, 1)) * 4 * np.pi - 2 * np.pi
            y = torch.sin(x)
            loss_val = loss(net(x), y)
            print(f"Iteration: {n+1}, Loss: {loss_val}")
