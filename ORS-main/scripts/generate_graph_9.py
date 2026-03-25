import torch

def process_tensor(x):
    return x.mean(), x.std(), x.max()

# 脚本化函数
scripted_add = torch.jit.script(process_tensor)

graph = scripted_add.graph

print(graph)
