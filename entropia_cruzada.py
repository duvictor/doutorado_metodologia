import torch
from torch import nn

criterion = nn.CrossEntropyLoss()
input = torch.tensor([[3.2, 1.3, 0.2, 0.8]], dtype=torch.float)
target = torch.tensor([0], dtype=torch.long)
print(criterion(input, target))