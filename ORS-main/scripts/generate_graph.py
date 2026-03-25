import torch

@torch.jit.script
def add_fn(x, y):
    return x + y

# 方式 1：导出原始 IR 文本
ir = add_fn.graph.str()  # 与 print(graph) 不同，这个是合法 IR！
print(ir)

# 方式 2：保存为 TorchScript 模型（可在 C++ 端加载）
add_fn.save("add_fn.pt")

