import torch
import torch.jit

def analyze_simple_add_graph():
    """详细分析 simple_add 函数的计算图"""
    
    def simple_add(x, y):
        return x + y

    # 脚本化函数
    scripted_add = torch.jit.script(simple_add)
    
    print("=== 详细计算图分析 ===")
    print("完整计算图:")
    print(scripted_add.graph)
    print("\n" + "="*50 + "\n")
    
    # 获取图对象
    graph = scripted_add.graph
    
    # 分析输入
    print("1. 图输入分析:")
    inputs = list(graph.inputs())
    for i, inp in enumerate(inputs):
        print(f"   输入 {i}: {inp}")
        print(f"     类型: {inp.type()}")
    
    # 分析节点
    print("\n2. 节点分析:")
    nodes = list(graph.nodes())
    for i, node in enumerate(nodes):
        print(f"   节点 {i}: {node}")
        print(f"     操作类型: {node.kind()}")
        print(f"     输入: {[str(inp) for inp in node.inputs()]}")
        print(f"     输出: {[str(out) for out in node.outputs()]}")
        
        # 分析节点属性
        if node.attributeNames():
            print(f"     属性: {node.attributeNames()}")
    
    # 分析输出
    print("\n3. 图输出分析:")
    outputs = list(graph.outputs())
    for i, out in enumerate(outputs):
        print(f"   输出 {i}: {out}")
        print(f"     类型: {out.type()}")
    
    # 显示生成的代码
    print("\n4. 生成的代码:")
    print(scripted_add.code)


    with open('./test_graph.txt', 'w', encoding='utf-8') as f:
        f.write(str(scripted_add.graph))

    
    return scripted_add

# 运行详细分析
scripted_add = analyze_simple_add_graph()
