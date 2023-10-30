import torch
import torch.nn as nn

# A Module whose forward method is simply a composition of Modules
# can be represented using the torch.nn.Sequential class
model = nn.Sequential(
    nn.Linear(3, 20),
    nn.ReLU(),
    nn.Linear(20, 30),
    nn.ReLU(),
    nn.Linear(30, 1),
)

# Prints a summary of the model architecture 
print(model)

x0 = torch.Tensor([1, 2, 3])
print(model(x0))