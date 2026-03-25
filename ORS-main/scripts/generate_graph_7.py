import torch

def analyze_complex_graph_no_if():
    """分析一个没有控制流的复杂 TorchScript 图"""

    def complex_fn(x, y):
        # step 1: 基础算子
        a = x + y
        b = x * y
        c = torch.matmul(a, b.T)  # 矩阵乘法

        # step 2: 激活与缩放
        d = torch.relu(c)
        e = torch.sigmoid(d) * 2.5

        # step 3: 归一化与拼接
        mean_val = torch.mean(e, dim=1, keepdim=True)
        std_val = torch.std(e, dim=1, keepdim=True)
        f = (e - mean_val) / (std_val + 1e-6)
        z = torch.cat([e, f], dim=1)

        return z

    # 脚本化
    scripted_fn = torch.jit.script(complex_fn)

    print("=== 无控制流的复杂计算图分析 ===")
    print("完整计算图:")
    print(scripted_fn.graph)
    print("\n" + "="*50 + "\n")

    graph = scripted_fn.graph

#    # 输入分析
#    print("1. 图输入:")
#    for i, inp in enumerate(graph.inputs()):
#        print(f"   输入 {i}: {inp}")
#        print(f"     类型: {inp.type()}")
#
#    # 节点分析
#    print("\n2. 节点分析:")
#    for i, node in enumerate(graph.nodes()):
#        print(f"   节点 {i}: {node}")
#        print(f"     操作类型: {node.kind()}")
#        print(f"     输入: {[str(inp) for inp in node.inputs()]}")
#        print(f"     输出: {[str(out) for out in node.outputs()]}")
#        if node.attributeNames():
#            attrs = {a: node[a] for a in node.attributeNames()}
#            print(f"     属性: {attrs}")
#
#    # 输出分析
#    print("\n3. 图输出:")
#    for i, out in enumerate(graph.outputs()):
#        print(f"   输出 {i}: {out}")
#        print(f"     类型: {out.type()}")
#
#    # 生成的TorchScript代码
#    print("\n4. 生成的TorchScript代码:")
#    print(scripted_fn.code)
#
#    # 保存图结构
    with open("./test_graph.txt", "w", encoding="utf-8") as f:
        f.write(str(scripted_fn.graph))

    return scripted_fn


if __name__ == "__main__":
    analyze_complex_graph_no_if()

