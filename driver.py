import torch
import torch.nn as nn
from torch.fx import symbolic_trace

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

model = TinyModel()

#converts model into graph
traced = symbolic_trace(model)

#prints graph
print(traced.graph)

#test it still works
x = torch.randn(1, 4)
print(traced(x))