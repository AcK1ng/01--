import torch
from torchvision.models import resnet18

def get_pure_operator_graph(model):
    # 必须 eval，否则 BatchNorm/Dropout 不能被折叠
    model = model.eval()

    # ① TorchScript 编译
    ts_model = torch.jit.script(model)

    # ② Freeze，会删除子模块并把 weight/bias 变成常量
    ts_model = torch.jit.freeze(ts_model)

    # ③ 取 inlined graph，而不是 forward() wrapper
    graph = ts_model.inlined_graph

    # ④ 强制 IR 优化 Pass
    torch._C._jit_pass_inline(graph)                  # 展开子图 (call_module → 算子)
    torch._C._jit_pass_constant_propagation(graph)    # 常量折叠
    torch._C._jit_pass_dce(graph)                     # 删除死节点
    torch._C._jit_pass_canonicalize(graph)            # 规范 IR
    torch._C._jit_pass_lint(graph)                    # 检查合法性

    return graph, ts_model


if __name__ == "__main__":
    model = resnet18()
    graph, ts_model = get_pure_operator_graph(model)

    print("=== ✅ Final Pure Operator Graph ===")
    print(graph)     # 不再包含 prim::GetAttr、call_module

    # 可选：保存到 .ir 文件
    with open("resnet18_aten.ir", "w") as f:
        f.write(str(graph))

    print("\nIR 已输出到 resnet18_aten.ir")
