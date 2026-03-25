import torch
from torchvision.models import resnet18
from torch._C import Graph

def extract_constant_tensors(graph: Graph):
    tensors = []
    for node in graph.nodes():
        if node.kind() == "prim::Constant" and node.hasAttribute("value"):
            val = node[attr::value]
            if isinstance(val, torch.Tensor):
                tensors.append(val)
    return tensors

if __name__ == "__main__":
    model = resnet18().eval()
    ts = torch.jit.freeze(torch.jit.script(model))
    graph = ts.inlined_graph

    torch._C._jit_pass_inline(graph)
    torch._C._jit_pass_constant_propagation(graph)

    const_tensors = extract_constant_tensors(graph)

    print(f"✅ 找到常量 Tensor 数量: {len(const_tensors)}")
    for i, t in enumerate(const_tensors[:10]):  # 只打印前 10 个
        print(f"[{i}] shape={tuple(t.shape)}, mean={t.float().mean():.4f}, sum={t.float().sum():.4f}")

    print("\n⚠️ 如果常量个数 ≈ 122 (resnet18 参数量)，说明权重已经成功嵌入 IR！")

