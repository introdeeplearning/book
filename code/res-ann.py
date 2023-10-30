import torch
import torch.nn as nn

class ResidualANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(3, 10)
        self.activation1 = nn.ReLU()
        self.affine2 = nn.Linear(10, 20)
        self.activation2 = nn.ReLU()
        self.affine3 = nn.Linear(20, 10)
        self.activation3 = nn.ReLU()
        self.affine4 = nn.Linear(10, 1)

    def forward(self, x0):
        x1 = self.activation1(self.affine1(x0))
        x2 = self.activation2(self.affine2(x1))
        x3 = self.activation3(x1 + self.affine3(x2))
        x4 = self.affine4(x3)
        return x4
        