import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.fx import symbolic_trace

from driver import apply_rewrite_rules, rules, benchmark, measure_memory


class CIFARModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


transform = transforms.Compose([
    transforms.ToTensor()
])

#
#test_loader = DataLoader(
#    test_dataset,
#    batch_size=512,
#    shuffle=False
#)

x, y = next(iter(test_loader))

model = CIFARModel()
model.eval()

traced = symbolic_trace(model)

original_output = traced(x)

rewritten_traced = apply_rewrite_rules(
    traced,
    rules
)

rewritten_output = rewritten_traced(x)

print("\nCIFAR OUTPUT CHECK:")
print(torch.allclose(original_output, rewritten_output, atol=1e-5))

print("\nCIFAR NODE COUNTS:")
print("Before:", len(list(symbolic_trace(model).graph.nodes)))
print("After: ", len(list(rewritten_traced.graph.nodes)))

original_time = benchmark(
    symbolic_trace(model),
    x,
    runs=300
)

rewritten_time = benchmark(
    rewritten_traced,
    x,
    runs=300
)

print("\nCIFAR BENCHMARK:")
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

print("\nCIFAR MEMORY:")
print("Original FX peak memory KB: ", original_memory)
print("Rewritten FX peak memory KB:", rewritten_memory)
print("Memory ratio:", original_memory / rewritten_memory)