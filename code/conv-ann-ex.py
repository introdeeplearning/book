import torch
import torch.nn as nn


model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(2, 2)),
    nn.ReLU(),
    nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1)),
)

with torch.no_grad():
    model[0].weight.set_(
        torch.Tensor([[[[0, 0], [0, 0]]], [[[1, 0], [0, 1]]]])
    )
    model[0].bias.set_(torch.Tensor([1, -1]))
    model[2].weight.set_(torch.Tensor([[[[-2]], [[2]]]]))
    model[2].bias.set_(torch.Tensor([3]))

x0 = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
print(model(x0))
