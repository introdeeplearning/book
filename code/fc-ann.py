import torch
import torch.nn as nn


class FullyConnectedANN(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the layers of the network in terms of Modules. 
        # nn.Linear(3, 20) represents an affine function defined 
        # by a 20x3 weight matrix and a 20-dimensional bias vector.
        self.affine1 = nn.Linear(3, 20)
        # The torch.nn.ReLU class simply wraps the 
        # torch.nn.functional.relu function as a Module.
        self.activation1 = nn.ReLU()
        self.affine2 = nn.Linear(20, 30)
        self.activation2 = nn.ReLU()
        self.affine3 = nn.Linear(30, 1)

    def forward(self, x0):
        x1 = self.activation1(self.affine1(x0))
        x2 = self.activation2(self.affine2(x1))
        x3 = self.affine3(x2)
        return x3


model = FullyConnectedANN()

x0 = torch.Tensor([1, 2, 3])
print(model(x0))

# Assigning a Module to an instance variable of a Module registers 
# all of the former's parameters as parameters of the latter
for p in model.parameters():
    print(p)