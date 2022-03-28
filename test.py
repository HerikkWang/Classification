import torch.nn as nn
import torch

a = torch.randn(5, 5, 3, 3)
b = nn.Softmax(dim=0)
c = b(a)
print(c[:, 1, 1, 1].sum())