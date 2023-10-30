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
lr = 0.0001
alpha = 0.9
beta = 0.999
eps = 1e-8
adj = 1.
adj2 = 1.

m = [p.clone().detach().zero_() for p in net.parameters()]
MM = [p.clone().detach().zero_() for p in net.parameters()]

for n in range(N):
    indices = torch.randint(0, M, (J,))

    x = X[indices]
    y = Y[indices]

    net.zero_grad()

    loss_val = loss(net(x), y)
    loss_val.backward()

    with torch.no_grad():
        adj *= alpha
        adj2 *= beta
        for m_p, M_p, p in zip(m, MM, net.parameters()):
            m_p.mul_(alpha)
            m_p.add_((1 - alpha) * p.grad)
            M_p.mul_(beta)
            M_p.add_((1 - beta) * p.grad * p.grad)
            p.sub_(lr * m_p / ((1 - adj) * (eps + (M_p / (1 - adj2)).sqrt())))

    if n % 1000 == 0:
        with torch.no_grad():
            x = torch.rand((1000, 1)) * 4 * np.pi - 2 * np.pi
            y = torch.sin(x)
            loss_val = loss(net(x), y)
            print(f"Iteration: {n+1}, Loss: {loss_val}")
