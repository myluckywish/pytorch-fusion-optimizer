import torch
import torch.nn as nn
from torch.fx import symbolic_trace

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

model = TinyModel()

traced = symbolic_trace(model)

print("FULL GRAPH:")
print(traced.graph)

print("\nNODES:")
for node in traced.graph.nodes:
    print(
        "name:", node.name,
        "| op:", node.op,
        "| target:", node.target,
        "| args:", node.args
    )

x = torch.randn(512, 4)
print(traced(x))

nodes = list(traced.graph.nodes)

print("\nPATTERN MATCHES:")

for i in range(len(nodes) - 1):
    current = nodes[i]
    nxt = nodes[i + 1]

    # Linear -> ReLU
    if current.target == "linear" and nxt.target == "relu":
        print("MATCH FOUND: Linear → ReLU")
        print("Possible transformation: fuse Linear and ReLU")

    # Linear -> Sigmoid
    if current.target == "linear" and nxt.target == "sigmoid":
        print("MATCH FOUND: Linear → Sigmoid")
        print("Possible transformation: fuse Linear and Sigmoid")

for i in range(len(nodes) - 2):
    a = nodes[i]
    b = nodes[i + 1]
    c = nodes[i + 2]

   
    if a.target == "conv" and b.target == "bn" and c.target == "relu":
        print("MATCH FOUND: Conv → BatchNorm → ReLU")
        print("Possible transformation: fuse Conv, BatchNorm, and ReLU")

  
    if a.target == "linear" and b.target == "bn" and c.target == "relu":
        print("MATCH FOUND: Linear → BatchNorm → ReLU")
        print("Possible transformation: fuse Linear, BatchNorm, and ReLU")

 
    if a.target == "linear" and b.target == "relu" and c.target == "linear":
        print("MATCH FOUND: Linear → ReLU → Linear")
        print("Possible transformation: mark as MLP block")