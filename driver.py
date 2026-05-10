import torch
import torch.nn as nn
from torch.fx import symbolic_trace


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(4, 16)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(16, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(16, 8)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(0.2)

        self.linear4 = nn.Linear(8, 4)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.bn1(x)
        x = self.relu2(x)

        x = self.linear3(x)
        x = self.sigmoid(x)

        x = self.dropout(x)

        x = self.linear4(x)
        return x


model = Model()
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


def is_module_type(node, traced, module_type):
    if node.op != "call_module":
        return False

    module = dict(traced.named_modules())[node.target]
    return isinstance(module, module_type)


x = torch.randn(512, 4)
print("\nMODEL OUTPUT:")
print(traced(x))

nodes = list(traced.graph.nodes)

print("\nPATTERN MATCHES:")

# 2-node patterns
for i in range(len(nodes) - 1):
    a = nodes[i]
    b = nodes[i + 1]

    if is_module_type(a, traced, nn.Linear) and is_module_type(b, traced, nn.ReLU):
        print("MATCH FOUND: Linear -> ReLU")
        print("Possible transformation: fuse Linear + ReLU\n")

    if is_module_type(a, traced, nn.Linear) and is_module_type(b, traced, nn.Sigmoid):
        print("MATCH FOUND: Linear -> Sigmoid")
        print("Possible transformation: fuse Linear + Sigmoid\n")

    if is_module_type(a, traced, nn.Linear) and is_module_type(b, traced, nn.Dropout):
        print("MATCH FOUND: Linear -> Dropout")
        print("Possible transformation: combine compute + regularization pass\n")


# 3-node patterns
for i in range(len(nodes) - 2):
    a = nodes[i]
    b = nodes[i + 1]
    c = nodes[i + 2]

    if (
        is_module_type(a, traced, nn.Linear)
        and is_module_type(b, traced, nn.BatchNorm1d)
        and is_module_type(c, traced, nn.ReLU)
    ):
        print("MATCH FOUND: Linear -> BatchNorm -> ReLU")
        print("Possible transformation: fuse Linear + BatchNorm + ReLU\n")

    if (
        is_module_type(a, traced, nn.Linear)
        and is_module_type(b, traced, nn.ReLU)
        and is_module_type(c, traced, nn.Linear)
    ):
        print("MATCH FOUND: Linear -> ReLU -> Linear")
        print("Possible transformation: mark as MLP block\n")

    if (
        is_module_type(a, traced, nn.Linear)
        and is_module_type(b, traced, nn.Sigmoid)
        and is_module_type(c, traced, nn.Linear)
    ):
        print("MATCH FOUND: Linear -> Sigmoid -> Linear")
        print("Possible transformation: activation block rewrite\n")