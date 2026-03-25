import torch
import torch.cuda.nvtx as nvtx
import time

def sync_all(devices):
    """
    辅助函数：强制同步列表中的所有 GPU。
    确保 GPU 0 的发送和 GPU 1 的计算都彻底完成。
    """
    for dev in devices:
        torch.cuda.synchronize(device=dev)

def verify_pipeline_and_profile():
    # ==========================================
    # 0. 环境准备与初始化
    # ==========================================
    if torch.cuda.device_count() < 2:
        print("Error: 此脚本需要至少 2 个 GPU。")
        return

    src_dev = torch.device('cuda:0')
    dst_dev = torch.device('cuda:1')
    device_list = [src_dev, dst_dev]

    print(f"Source Device: {src_dev}")
    print(f"Dest Device:   {dst_dev}")

    # 创建 4 个流，实现计算与通信分离
    src_compute = torch.cuda.Stream(device=src_dev)
    src_tx      = torch.cuda.Stream(device=src_dev) # 负责发送
    dst_rx      = torch.cuda.Stream(device=dst_dev) # 负责信号中继
    dst_compute = torch.cuda.Stream(device=dst_dev)

    # 数据初始化 (N=4096 用于产生足够的负载)
    N = 4096
    print(f"初始化 Tensor ({N}x{N})...")
    with torch.cuda.stream(src_compute):
        data_src = torch.randn(N, N, device=src_dev)

    # 预热 CUDA Context (避免冷启动开销影响 Trace)
    print("正在预热 CUDA...")
    for _ in range(3):
        torch.mm(data_src, data_src)
    sync_all(device_list)

    print("\n>>> 开始执行 Profiling (请使用 nsys 查看 Trace) <<<\n")

    # ==========================================
    # Phase 1: 实验组 - 多流流水线并行
    # ==========================================
    
    # [NVTX] 标记整个实验阶段
    nvtx.range_push(">>> PHASE 1: PIPELINE PARALLELISM (Optimized) <<<")

    # --- Step 1: 源端计算 (GPU 0 Compute) ---
    with torch.cuda.stream(src_compute):
        nvtx.range_push("1. Src Compute")
        # 模拟产生数据: B = A * A
        result_src = torch.matmul(data_src, data_src)
        
        # [Event] 记录：数据 B 生产完毕
        event_comp_done = torch.cuda.Event(blocking=False)
        event_comp_done.record(src_compute)
        nvtx.range_pop()

        # [验证重叠] 模拟不依赖数据的后续计算
        # 在 Nsight 中，这应该与 Step 2 的 Copy 并行
        nvtx.range_push("1.5. Overlap Check")
        dummy_compute = torch.matmul(data_src, data_src)
        nvtx.range_pop()

    # --- Step 2: 源端发送 (GPU 0 Tx) ---
    with torch.cuda.stream(src_tx):
        nvtx.range_push("2. Tx Wait & Copy")
        
        # [Wait] Tx 等待 Comp：防止读到旧数据 (Read-After-Write 保护)
        src_tx.wait_event(event_comp_done)
        
        # [Copy] P2P 传输 (必须 non_blocking=True)
        # 注意：Tensor.to 会提交到当前的 src_tx 流
        nvtx.range_push("2.5 result_src.to")
        data_dst_buffer = result_src.to(dst_dev, non_blocking=True)
        nvtx.range_pop()
        
        # [Event] 记录：发送动作结束
        event_tx_done = torch.cuda.Event(blocking=False)
        event_tx_done.record(src_tx)
        nvtx.range_pop()

    # --- Step 3: 目标端接收中继 (GPU 1 Rx) ---
    with torch.cuda.stream(dst_rx):
        nvtx.range_push("3. Rx Signal Relay")
        
        # [Wait] Rx 跨设备等待 Tx：确保数据已落地
        dst_rx.wait_event(event_tx_done)
        
        # [Event] 记录：数据已就绪，通知计算流
        event_rx_ready = torch.cuda.Event(blocking=False)
        event_rx_ready.record(dst_rx)
        nvtx.range_pop()

    # --- Step 4: 目标端计算 (GPU 1 Compute) ---
    with torch.cuda.stream(dst_compute):
        nvtx.range_push("4. Dst Compute")
        
        # [Wait] Comp 等待 Rx
        dst_compute.wait_event(event_rx_ready)
        
        # 使用传输过来的数据进行计算: C = B * B
        final_result = torch.matmul(data_dst_buffer, data_dst_buffer)
        nvtx.range_pop()

    # [关键] 显式同步所有设备，确保 Phase 1 彻底结束
    # 否则 Nsight 可能会截断 GPU 1 的最后一段计算
    sync_all(device_list)
    
    # [NVTX] 结束实验阶段
    nvtx.range_pop()

    # 插入一个明显的标记
    nvtx.mark("--- BARRIER: SWITCHING TO VERIFICATION ---")

    # ==========================================
    # Phase 2: 对照组 - 串行逻辑验证 (Ground Truth)
    # ==========================================
    
    # [NVTX] 标记验证阶段
    nvtx.range_push(">>> PHASE 2: GROUND TRUTH (Sequential) <<<")

    # 使用默认流，串行执行，作为标准答案
    # 逻辑：A -> A*A -> Copy -> (A*A)*(A*A)
    
    nvtx.range_push("GT: Compute Src")
    expected_B = torch.matmul(data_src, data_src)
    nvtx.range_pop()
    
    nvtx.range_push("GT: Copy")
    expected_B_dst = expected_B.to(dst_dev) # 默认同步拷贝
    nvtx.range_pop()
    
    nvtx.range_push("GT: Compute Dst")
    expected_C = torch.matmul(expected_B_dst, expected_B_dst)
    nvtx.range_pop()

    # 再次同步所有设备
    sync_all(device_list)
    
    # [NVTX] 结束验证阶段
    nvtx.range_pop()

    print("Profiling 阶段结束。\n")

    # ==========================================
    # 3. 结果数值比对
    # ==========================================
    print(">>> 正在进行数值验证 <<<")
    
    # 允许一定的浮点误差 (float32 累积误差)
    is_close = torch.allclose(final_result, expected_C, atol=1e-2, rtol=1e-3)
    
    if is_close:
        print("✅ 验证成功 (PASSED)")
        print("流水线并行结果与串行基准一致。依赖关系逻辑正确。")
    else:
        print("❌ 验证失败 (FAILED)")
        max_diff = (final_result - expected_C).abs().max().item()
        print(f"最大误差: {max_diff}")
        print("原因分析: 可能是Event同步缺失导致目标端读取了未完成的数据。")

if __name__ == "__main__":
    verify_pipeline_and_profile()
