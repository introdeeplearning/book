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

for n in range(N):
    indices = torch.randint(0, M, (J,))

    x = X[indices]
    y = Y[indices]

    net.zero_grad()

    # Remember the original parameters
    params = [p.clone().detach() for p in net.parameters()]
    # Compute the loss
    loss_val = loss(net(x), y)
    # Compute the gradients with respect to the parameters
    loss_val.backward()

    with torch.no_grad():
        # Make a half-step in the direction of the negative 
        # gradient
        for p in net.parameters():
            if p.grad is not None:
                p.sub_(0.5 * lr * p.grad)

    net.zero_grad()
    # Compute the loss and the gradients at the midpoint
    loss_val = loss(net(x), y)
    loss_val.backward()

    with torch.no_grad():
        # Subtract the scaled gradient at the midpoint from the
        # original parameters
        for param, midpoint_param in zip(
            params, net.parameters()
        ):
            param.sub_(lr * midpoint_param.grad)

        # Copy the new parameters into the model
        for param, p in zip(params, net.parameters()):
            p.copy_(param)

    if n % 1000 == 0:
        with torch.no_grad():
            x = torch.rand((1000, 1)) * 4 * np.pi - 2 * np.pi
            y = torch.sin(x)
            loss_val = loss(net(x), y)
            print(f"Iteration: {n+1}, Loss: {loss_val}")
