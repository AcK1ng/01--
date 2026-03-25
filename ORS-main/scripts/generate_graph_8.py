import torch

def simple_add_mul(x, y):
    c = torch.cat([x, y], dim=0)  # cat
    return c

scripted = torch.jit.script(simple_add_mul)
print(scripted.graph)

