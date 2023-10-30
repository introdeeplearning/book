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
lr = 0.003
alpha = 0.999

m = [p.clone().detach().zero_() for p in net.parameters()]

for n in range(N):
    indices = torch.randint(0, M, (J,))

    x = X[indices]
    y = Y[indices]

    net.zero_grad()

    # Remember the original parameters
    params = [p.clone().detach() for p in net.parameters()]

    for p, m_p in zip(params, m):
        p.sub_(lr * alpha * m_p)

    # Compute the loss
    loss_val = loss(net(x), y)
    # Compute the gradients with respect to the parameters
    loss_val.backward()

    with torch.no_grad():
        for p, m_p, q in zip(net.parameters(), m, params):
            m_p.mul_(alpha)
            m_p.add_((1 - alpha) * p.grad)
            q.sub_(lr * m_p)
            p.copy_(q)

    if n % 1000 == 0:
        with torch.no_grad():
            x = torch.rand((1000, 1)) * 4 * np.pi - 2 * np.pi
            y = torch.sin(x)
            loss_val = loss(net(x), y)
            print(f"Iteration: {n+1}, Loss: {loss_val}")
