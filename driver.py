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


class FusedLinearReLU(nn.Module):
    def __init__(self, linear, relu):
        super().__init__()
        self.linear = linear
        self.relu = relu

    def forward(self, x):
        return self.relu(self.linear(x))


def is_module_type(node, traced, module_type):
    if node.op != "call_module":
        return False

    module = dict(traced.named_modules())[node.target]
    return isinstance(module, module_type)


def fuse_linear_relu(traced):
    modules = dict(traced.named_modules())
    graph = traced.graph

    nodes = list(graph.nodes)
    fusion_count = 0

    for i in range(len(nodes) - 1):
        linear_node = nodes[i]
        relu_node = nodes[i + 1]

        if not (
            is_module_type(linear_node, traced, nn.Linear)
            and is_module_type(relu_node, traced, nn.ReLU)
        ):
            continue

        if relu_node.args[0] != linear_node:
            continue

        fusion_count += 1
        fused_name = f"fused_linear_relu_{fusion_count}"

        linear_module = modules[linear_node.target]
        relu_module = modules[relu_node.target]

        fused_module = FusedLinearReLU(linear_module, relu_module)

        traced.add_module(fused_name, fused_module)

        with graph.inserting_after(relu_node):
            fused_node = graph.call_module(fused_name, args=linear_node.args)

        relu_node.replace_all_uses_with(fused_node)

        graph.erase_node(relu_node)
        graph.erase_node(linear_node)

    graph.lint()
    traced.recompile()

    return traced


model = Model()
traced = symbolic_trace(model)

print("BEFORE GRAPH:")
print(traced.graph)

x = torch.randn(512, 4)

original_output = traced(x)

fused_traced = fuse_linear_relu(traced)

print("\nAFTER GRAPH:")
print(fused_traced.graph)

fused_output = fused_traced(x)

print("\nOUTPUT SHAPES:")
print("Original output shape:", original_output.shape)
print("Fused output shape:   ", fused_output.shape)

print("\nOUTPUT CHECK:")
print("Outputs close:", torch.allclose(original_output, fused_output, atol=1e-6))

print("\nMODULES AFTER FUSION:")
for name, module in fused_traced.named_modules():
    if name != "":
        print(name, "->", module.__class__.__name__)