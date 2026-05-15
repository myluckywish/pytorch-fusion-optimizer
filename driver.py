import torch
import torch.nn as nn
from torch.fx import symbolic_trace
import time
import tracemalloc

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


# generic fusion block
class SequentialFusion(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class RewriteRule:
    def __init__(self, name, pattern, replacement):
        self.name = name
        self.pattern = pattern
        self.replacement = replacement


def is_module_type(node, traced, module_type):
    if node.op != "call_module":
        return False

    module = dict(traced.named_modules())[node.target]

    return isinstance(module, module_type)


def matches_pattern(nodes, start, traced, pattern):
    if start + len(pattern) > len(nodes):
        return False

    for offset, module_type in enumerate(pattern):
        node = nodes[start + offset]

        if not is_module_type(node, traced, module_type):
            return False

        # verify graph connectivity
        if offset > 0:
            prev_node = nodes[start + offset - 1]

            if node.args[0] != prev_node:
                return False

    return True


def apply_rewrite_rules(traced, rules):
    graph = traced.graph

    fusion_count = 0
    changed = True

    while changed:
        changed = False

        modules = dict(traced.named_modules())
        nodes = list(graph.nodes)

        for rule in rules:

            for i in range(len(nodes)):

                if not matches_pattern(
                    nodes,
                    i,
                    traced,
                    rule.pattern
                ):
                    continue

                matched_nodes = nodes[i:i + len(rule.pattern)]

                first_node = matched_nodes[0]
                last_node = matched_nodes[-1]

                old_modules = [
                    modules[node.target]
                    for node in matched_nodes
                ]

                fusion_count += 1

                fused_name = f"fused_{rule.name}_{fusion_count}"

                # dynamically create fusion block
                fused_module = rule.replacement(*old_modules)

                traced.add_module(
                    fused_name,
                    fused_module
                )

                with graph.inserting_after(last_node):

                    fused_node = graph.call_module(
                        fused_name,
                        args=first_node.args
                    )

                last_node.replace_all_uses_with(fused_node)

                for node in reversed(matched_nodes):
                    graph.erase_node(node)

                graph.lint()
                traced.recompile()

                changed = True
                break

            if changed:
                break

    return traced


def benchmark(model, x, runs=1000):
    model.eval()

    # warmup
    for _ in range(50):
        model(x)

    start = time.perf_counter()

    for _ in range(runs):
        model(x)

    end = time.perf_counter()

    return (end - start) / runs

def measure_memory(model, x):
    model.eval()

    tracemalloc.start()

    model(x)

    current, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    return peak / 1024
    
rules = [

    RewriteRule(
        name="linear_relu",
        pattern=[nn.Linear, nn.ReLU],
        replacement=SequentialFusion
    ),

    RewriteRule(
        name="linear_sigmoid",
        pattern=[nn.Linear, nn.Sigmoid],
        replacement=SequentialFusion
    ),

    RewriteRule(
        name="linear_bn_relu",
        pattern=[
            nn.Linear,
            nn.BatchNorm1d,
            nn.ReLU
        ],
        replacement=SequentialFusion
    ),
]


model = Model()
model.eval()

traced = symbolic_trace(model)

print("BEFORE GRAPH:")
print(traced.graph)

x = torch.randn(512, 4)

original_output = traced(x)

rewritten_traced = apply_rewrite_rules(
    traced,
    rules
)

print("\nAFTER GRAPH:")
print(rewritten_traced.graph)

rewritten_output = rewritten_traced(x)

print("\nOUTPUT CHECK:")
print(
    torch.allclose(
        original_output,
        rewritten_output,
        atol=1e-6
    )
)

print("\nMODULES AFTER REWRITE:")

for name, module in rewritten_traced.named_modules():

    if name != "":
        print(name, "->", module.__class__.__name__)


print("\nNODE COUNTS:")

before_nodes = len(list(symbolic_trace(model).graph.nodes))
after_nodes = len(list(rewritten_traced.graph.nodes))

print("Before:", before_nodes)
print("After: ", after_nodes)


original_time = benchmark(
    symbolic_trace(model),
    x
)

rewritten_time = benchmark(
    rewritten_traced,
    x
)

print("\nBENCHMARK:")
print("Original FX avg time: ", original_time)
print("Rewritten FX avg time:", rewritten_time)
print("Speedup:", original_time / rewritten_time)

original_memory = measure_memory(
    symbolic_trace(model),
    x
)

rewritten_memory = measure_memory(
    rewritten_traced,
    x
)

print("\nMEMORY:")
print("Original FX peak memory KB: ", original_memory)
print("Rewritten FX peak memory KB:", rewritten_memory)
print("Memory ratio:", original_memory / rewritten_memory)