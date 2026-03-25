import torch
import sys
from graphviz import Digraph

# 定义需要忽略的“噪音”算子类型
# 这些算子通常不影响理解网络主干结构，但会极大增加绘图负担
IGNORE_OPS = {
    'prim::Constant', 
    'prim::ListConstruct', 
    'prim::TupleIndex',
    'prim::TupleConstruct',
    'prim::GetAttr',
    'prim::If',      # If分支如果不做特殊处理会让图很乱，暂时忽略
    'prim::Loop',    # 循环同理
    'aten::size',    # 获取形状，不影响数据流
    'aten::len',
    'aten::Int',
}

def render_ir_hierarchical(ir_text, output_filename="ir_hierarchy"):
    try:
        graph = torch.parse_ir(ir_text)
    except Exception as e:
        print(f"解析失败: {e}")
        return

    # 1. 强制使用 'dot' 引擎以保证从上到下的层级
    # 2. 关键属性优化：
    #    rankdir='TB': Top to Bottom
    #    splines='polyline': 使用折线代替曲线，减少缠绕
    #    nodesep/ranksep: 增加节点和层级之间的距离
    #    concentrate='true': 合并重叠的线
    dot = Digraph(format='svg', engine='dot')
    dot.attr(rankdir='TB', splines='polyline', nodesep='0.5', ranksep='1.0', concentrate='true')
    
    # 节点样式
    dot.attr('node', shape='box', style='filled', fillcolor='aliceblue', fontname='Helvetica')

    # 映射：Value -> Producer Node Name
    val_to_producer = {}

    # --- 1. 处理输入 ---
    # 使用 subgraph 保证输入在最顶层
    with dot.subgraph(name='cluster_0_inputs') as c:
        c.attr(style='invis') # 隐藏边框
        for i, inp_val in enumerate(graph.inputs()):
            node_name = f"in_{i}"
            label = f"Input\n%{inp_val.debugName()}"
            c.node(node_name, label=label, shape='oval', fillcolor='gold')
            val_to_producer[inp_val] = node_name

    # --- 2. 处理中间节点 (带过滤) ---
    # 限制节点数量：如果图太大，只画前 500 个节点用于调试
    # 如果你想画全图，把下面的 LIMIT 改大，但风险是跑不完
    LIMIT = 500
    
    count = 0
    for i, node in enumerate(graph.nodes()):
        kind = node.kind()
        
        # [核心优化]：如果是在黑名单里的算子，直接跳过不画
        if kind in IGNORE_OPS:
            continue
            
        count += 1
        if count > LIMIT:
            print(f"警告: 节点数超过 {LIMIT}，已停止绘制剩余部分以防止卡死。")
            break

        # 创建节点 ID
        op_node_name = f"op_{i}"
        
        # 生成标签：去掉冗长的命名空间，只留关键名
        # 例如: aten::matmul -> matmul
        short_name = kind.split('::')[-1] 
        
        # 如果是计算密集型算子，加深颜色
        color = 'aliceblue'
        if 'matmul' in short_name or 'conv' in short_name or 'linear' in short_name:
            color = 'coral'
        elif 'add' in short_name or 'mul' in short_name:
            color = 'lightgreen'

        dot.node(op_node_name, label=short_name, fillcolor=color)

        # 记录输出
        for out_val in node.outputs():
            val_to_producer[out_val] = op_node_name

        # [连线逻辑]
        # 只连接那些“被画出来的”节点
        # 如果输入的源头被我们过滤掉了（比如是 Constant），这条线就不画了
        # 这样能极大简化图的连线复杂度
        added_edges = set()
        for inp_val in node.inputs():
            if inp_val in val_to_producer:
                producer = val_to_producer[inp_val]
                
                # 防止重复连线
                if producer not in added_edges:
                    dot.edge(producer, op_node_name)
                    added_edges.add(producer)

    # --- 3. 处理输出 ---
    with dot.subgraph(name='cluster_99_outputs') as c:
        c.attr(style='invis')
        for i, out_val in enumerate(graph.outputs()):
            out_node_name = f"out_{i}"
            c.node(out_node_name, label="Output", shape='doubleoctagon', fillcolor='gold')
            
            if out_val in val_to_producer:
                producer = val_to_producer[out_val]
                dot.edge(producer, out_node_name)

    print(f"正在渲染 {count} 个节点的层级图，请稍候...")
    # view=False 防止服务器端弹窗报错
    output_path = dot.render(output_filename, view=False)
    print(f"渲染完成！文件保存为: {output_path}")

# 使用方式
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 读取你的 txt 文件
        with open(sys.argv[1], 'r') as f:
            content = f.read()
        render_ir_hierarchical(content, sys.argv[1] + "_hierarchical")
    else:
        print("请提供文件路径")
