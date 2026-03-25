import torch

def export_script_graph():
    # 1. 定义计算逻辑
    # 使用 @torch.jit.script 装饰器，或者后续用 torch.jit.script() 调用
    # Type Hint (如 : torch.Tensor) 对 JIT 解析很有帮助
    @torch.jit.script
    def pipeline_add_task(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # Node 1: C = A + B
        # 你的系统可以将此节点调度到 Device 0
        c = a + b
        
        # Node 2: E = C + D
        # 你的系统可以将此节点调度到 Device 1
        # 此时你的系统需要检测到 c 和 d 在不同设备，并在两者之间插入 Event Wait 和 Copy
        e = c + d
        
        return e

    # 2. 打印计算图结构 (IR)
    print(">>> JIT Graph IR (中间表示) <<<")
    # 这展示了底层的节点和数据流
    print(pipeline_add_task.graph)
    with open("./test_graph.txt", "w", encoding="utf-8") as f:
        f.write(str(pipeline_add_task.graph))

    # 3. 打印可读代码
    print("\n>>> JIT Readable Code <<<")
    print(pipeline_add_task.code)

    # 4. 保存为 TorchScript 文件
    # 这样你的 C++ 调度系统就可以通过 torch::jit::load() 加载它
    model_path = "pipeline_add_graph.pt"
    pipeline_add_task.save(model_path)
    print(f"\n✅ 模型已保存至: {model_path}")
    print("你的 C++ 系统可以加载此文件并遍历 Graph 进行节点调度。")

if __name__ == "__main__":
    export_script_graph()
