import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

M = 10000  # number of training samples

# We fix a random seed. This is not necessary for training a
# neural network, but we use it here to ensure that the same
# plot is created on every run.
torch.manual_seed(0)

# Here, we define the training set.
# Create a tensor of shape (M, 1) with entries sampled from a
# uniform distribution on [-2 * pi, 2 * pi)
X = (torch.rand((M, 1)) - 0.5) * 4 * np.pi
# We use the sine as the target function, so this defines the 
# desired outputs.
Y = torch.sin(X)

J = 32  # the batch size
N = 100000  # the number of SGD iterations

loss = nn.MSELoss()  # the mean squared error loss function
gamma = 0.003  # the learning rate

# Define a network with a single hidden layer of 200 neurons and 
# tanh activation function
net = nn.Sequential(
    nn.Linear(1, 200), nn.Tanh(), nn.Linear(200, 1)
)

# Set up a 3x3 grid of plots
fig, axs = plt.subplots(
    3,
    3,
    figsize=(12, 8),
    sharex="col",
    sharey="row",
)

# Plot the target function
x = torch.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape((1000, 1))
y = torch.sin(x)
for ax in axs.flatten():
    ax.plot(x, y, label="Target")
    ax.set_xlim([-2 * np.pi, 2 * np.pi])
    ax.set_ylim([-1.1, 1.1])

plot_after = [1, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]

# The training loop
for n in range(N):
    # Choose J samples randomly from the training set
    indices = torch.randint(0, M, (J,))
    X_batch = X[indices]
    Y_batch = Y[indices]

    net.zero_grad()  # Zero out the gradients

    loss_val = loss(net(X_batch), Y_batch)  # Compute the loss
    loss_val.backward()  # Compute the gradients

    # Update the parameters
    with torch.no_grad():
        for p in net.parameters():
            # Subtract the scaled gradient in-place
            p.sub_(gamma * p.grad)

    if n + 1 in plot_after:
        # Plot the realization function of the ANN
        i = plot_after.index(n + 1)
        ax = axs[i // 3][i % 3]
        ax.set_title(f"Batch {n+1}")

        with torch.no_grad():
            ax.plot(x, net(x), label="ANN realization")

axs[0][0].legend(loc="upper right")

plt.tight_layout()
plt.savefig("../../plots/sgd.pdf", bbox_inches="tight")
