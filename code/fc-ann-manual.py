import torch
import torch.nn as nn
import torch.nn.functional as F


# To define a neural network, we define a class that inherits from 
# torch.nn.Module
class FullyConnectedANN(nn.Module):
    def __init__(self):
        super().__init__()
        # In the constructor, we define the weights and biases.
        # Wrapping the tensors in torch.nn.Parameter objects tells 
        # PyTorch that these are parameters that should be 
        # optimized during training.
        self.W1 = nn.Parameter(
            torch.Tensor([[1, 0], [0, -1], [-2, 2]])
        )
        self.B1 = nn.Parameter(torch.Tensor([0, 2, -1]))
        self.W2 = nn.Parameter(torch.Tensor([[1, -2, 3]]))
        self.B2 = nn.Parameter(torch.Tensor([1]))

    # The realization function of the network
    def forward(self, x0):
        x1 = F.relu(self.W1 @ x0 + self.B1)
        x2 = self.W2 @ x1 + self.B2
        return x2


model = FullyConnectedANN()

x0 = torch.Tensor([1, 2])
# Print the output of the realization function for input x0
print(model.forward(x0))

# As a consequence of inheriting from torch.nn.Module we can just 
# "call" the model itself (which will call the forward method 
# implicitly)
print(model(x0))

# Wrapping a tensor in a Parameter object and assigning it to an 
# instance variable of the Module makes PyTorch register it as a 
# parameter. We can access all parameters via the parameters 
# method.
for p in model.parameters():
    print(p)
