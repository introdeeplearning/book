import torch
import matplotlib.pyplot as plt

# Use the GPU if available
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Computes an approximation of E[|phi(sqrt(2*rho*T) W + xi) -
# N(xi)|²] with W a standard normal random variable using the rows
# of x as # independent realizations of the random variable xi
def loss(N, rho, phi, t, x):
    W = torch.randn_like(x).to(dev)
    return (phi(torch.sqrt(2 * rho * t) * W + x) - 
            N(torch.cat((t,x),1))).square().mean()

d = 2              # the input dimension
a, b = -5.0, 5.0   # the domain will be [a,b]^d
T = 2.0            # the time horizon
rho = 1.0          # the diffusivity

# Define the initial value
def phi(x):
    return x.cos().sum(axis=1, keepdim=True)

# Define a neural network with two hidden layers with 50 neurons 
# each using ReLU activations
N = torch.nn.Sequential(
    torch.nn.Linear(d+1, 50), torch.nn.ReLU(),
    torch.nn.Linear(50, 50), torch.nn.ReLU(),
    torch.nn.Linear(50, 1)
).to(dev)

# Configure the training parameters and optimization algorithm
steps = 3000
batch_size = 256
optimizer = torch.optim.Adam(N.parameters())

# Train the network
for step in range(steps):
    # Generate uniformly distributed samples from [a,b]^d
    x = (torch.rand(batch_size, d) * (b-a) + a).to(dev)
    t = T * torch.rand(batch_size, 1).to(dev)
    
    optimizer.zero_grad()
    # Compute the loss
    L = loss(N, rho, phi, t, x)
    # Compute the gradients
    L.backward()
    # Apply changes to weights and biases of N
    optimizer.step()

# Plot the result at M+1 timesteps
M = 5
mesh = 128

def toNumpy(t):
    return t.detach().cpu().numpy().reshape((mesh,mesh))

fig, axs = plt.subplots(2,3,subplot_kw=dict(projection='3d'))
fig.set_size_inches(16, 10)
fig.set_dpi(300)

for i in range(M+1):
    x = torch.linspace(a, b, mesh)
    y = torch.linspace(a, b, mesh)
    x, y = torch.meshgrid(x, y, indexing='xy')
    x = x.reshape((mesh*mesh,1)).to(dev)
    y = y.reshape((mesh*mesh,1)).to(dev)
    z = N(torch.cat((i*T/M*torch.ones(128*128,1).to(dev), x, y), 
                    1))

    axs[i//3,i%3].set_title(f"t = {i * T / M}")
    axs[i//3,i%3].set_zlim(-2,2)
    axs[i//3,i%3].plot_surface(toNumpy(x), toNumpy(y), toNumpy(z), 
                    cmap='viridis')

fig.savefig(f"../plots/kolmogorov.pdf", bbox_inches='tight')