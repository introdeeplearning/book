import torch
import torch.nn as nn


class ConvolutionalANN(nn.Module):
    def __init__(self):
        super().__init__()
        # The convolutional layer defined here takes any tensor of 
        # shape (1, n, m) [a single input] or (N, 1, n, m) [a batch 
        # of N inputs] where N, n, m are natural numbers satisfying 
        # n >= 3 and m >= 3.
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=5, kernel_size=(3, 3)
        )
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=5, out_channels=5, kernel_size=(5, 3)
        )

    def forward(self, x0):
        x1 = self.activation1(self.conv1(x0))
        print(x1.shape)
        x2 = self.conv2(x1)
        print(x2.shape)
        return x2


model = ConvolutionalANN()
x0 = torch.rand(1, 20, 20)
# This will print the shapes of the outputs of the two layers of
# the model, in this case:
# torch.Size([5, 18, 18])
# torch.Size([5, 14, 16])
model(x0)
