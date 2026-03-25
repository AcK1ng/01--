import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

def verify_scheme():
    # 0. 环境检查
    if torch.cuda.device_count() < 2:
        print("Error: 需要至少两个 GPU 才能运行此验证程序。")
        return

    src_dev = torch.device('cuda:0')
    dst_dev = torch.device('cuda:1')
    
    print(f"Source: {src_dev}, Destination: {dst_dev}")

    # 1. 创建流 (Streams)
    # GPU 0: 计算流 & 传输流 (Tx)
    src_compute_stream = torch.cuda.Stream(device=src_dev)
    src_tx_stream = torch.cuda.Stream(device=src_dev)

    # GPU 1: 接收流 (Rx) & 计算流
    dst_rx_stream = torch.cuda.Stream(device=dst_dev)
    dst_compute_stream = torch.cuda.Stream(device=dst_dev)

    # 准备数据 (矩阵要足够大，才能在 Profiler 里看清楚)
    N = 4096
    # 在 GPU 0 上创建数据
    with torch.cuda.stream(src_compute_stream):
        data_src = torch.randn(N, N, device=src_dev)

    # 预热 CUDA Context
    print("正在预热 CUDA...")
    for _ in range(3):
        torch.mm(data_src, data_src)
    torch.cuda.synchronize()

    print("开始执行并记录 Trace...")

    # 使用 Profiler 记录时间轴
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        
        # === 步骤 1: 源设备计算 (GPU 0 Compute Stream) ===
        with torch.cuda.stream(src_compute_stream):
            with record_function("1. Source Compute (MatMul)"):
                # 模拟一个耗时的算子
                result_src = torch.matmul(data_src, data_src)
            
            # [关键点]: 插入 Event，表示数据生产完毕
            event_comp_done = torch.cuda.Event(blocking=False)
            event_comp_done.record(src_compute_stream)

            # [验证 Overlap]: 在计算流继续做一个不依赖该数据的任务
            # 如果方案有效，这个任务应该和下面的 copy 并行执行
            with record_function("X. Overlapped Compute (Source)"):
                dummy = torch.matmul(data_src, data_src)

        # === 步骤 2: 源设备发送 (GPU 0 Tx Stream) ===
        with torch.cuda.stream(src_tx_stream):
            with record_function("2. Tx Wait & Copy"):
                # [关键点]: Tx 流等待 Compute 流
                src_tx_stream.wait_event(event_comp_done)
                
                # 执行 P2P 拷贝
                # 注意: non_blocking=True 是必须的，否则会强制同步 CPU
                # 此时 .to() 操作被提交到了 src_tx_stream
                data_dst_buffer = result_src.to(dst_dev, non_blocking=True)
            
            # [关键点]: 插入 Event，表示发送完毕
            event_tx_done = torch.cuda.Event(blocking=False)
            event_tx_done.record(src_tx_stream)

        # === 步骤 3: 目标设备接收 (GPU 1 Rx Stream) ===
        with torch.cuda.stream(dst_rx_stream):
            # 注意：这里我们切换到了 cuda:1 的流
            # Rx 流充当信号中继
            with record_function("3. Rx Signal Wait"):
                # [关键点]: 跨设备等待 (GPU 1 等 GPU 0)
                dst_rx_stream.wait_event(event_tx_done)
            
            # [关键点]: 插入 Event，表示 Rx 侧已确信数据到达
            event_rx_ready = torch.cuda.Event(blocking=False)
            event_rx_ready.record(dst_rx_stream)

        # === 步骤 4: 目标设备计算 (GPU 1 Compute Stream) ===
        with torch.cuda.stream(dst_compute_stream):
            with record_function("4. Dst Compute Wait & Execute"):
                # [关键点]: 目标计算流等待 Rx 流
                dst_compute_stream.wait_event(event_rx_ready)
                
                # 使用传输过来的数据进行计算
                final_result = torch.matmul(data_dst_buffer, data_dst_buffer)

    # 同步所有流以确保执行完成
    torch.cuda.synchronize()
    print("执行完成。正在导出 trace.json ...")
    
    # 导出文件，可以使用 Chrome 浏览器打开 chrome://tracing 查看，或者用 VSCode TensorBoard
    prof.export_chrome_trace("trace_stream_overlap.json")
    print("Trace 文件已生成: trace_stream_overlap.json")

    # 简单的正确性验证
    print("正在验证计算结果正确性...")
    expected = torch.matmul(torch.matmul(data_src, data_src).to(dst_dev), torch.matmul(data_src, data_src).to(dst_dev))
    # 允许一点浮点误差
    if torch.allclose(final_result, expected, atol=1e-3):
        print("✅ 验证成功：结果数据正确，依赖关系处理无误！")
    else:
        print("❌ 验证失败：结果数据不一致，可能是同步逻辑缺失导致读到了脏数据。")

if __name__ == "__main__":
    verify_scheme()
