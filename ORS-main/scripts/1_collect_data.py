import torch
import numpy as np
import pandas as pd
import os

def collect_data(src_dev='cuda:0', dst_dev='cuda:1'):
    print(f"=== 开始采集数据: {src_dev} -> {dst_dev} ===")
    print("这可能需要几分钟，请耐心等待...")

    # 1. 生成采样点 (Size in Bytes)
    # 策略：
    # - 小数据区 (1KB - 128KB): 线性密集采样，为了看清 Latency
    # - 中间区 (128KB - 4MB): 对数采样，为了捕捉 Threshold
    # - 大数据区 (4MB - 256MB): 对数采样，为了拟合 Bandwidth
    
    # 避免重复点并排序
    sizes = np.unique(np.concatenate([
        np.linspace(1024, 128*1024, 50),           # 50个点
        np.geomspace(128*1024, 4*1024*1024, 40),   # 40个点
        np.geomspace(4*1024*1024, 256*1024*1024, 30) # 30个点
    ])).astype(int)

    # 2. 准备 GPU 资源
    # 分配最大显存作为池子，避免循环中反复 malloc
    max_size = sizes.max()
    try:
        src_tensor = torch.randn(max_size // 4 + 1024, device=src_dev, dtype=torch.float32)
    except RuntimeError as e:
        print(f"Error: 显存不足，无法分配 {max_size/1024/1024:.2f} MB")
        return

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    data_list = []

    # 3. 开始循环测试
    total_steps = len(sizes)
    for i, size in enumerate(sizes):
        if i % 10 == 0:
            print(f"Progress: {i}/{total_steps} (Current Size: {size/1024:.2f} KB)")

        # View 操作是零开销的
        sub_tensor = src_tensor[:(size // 4)]
        
        # Warmup (非常重要，建立页表映射)
        for _ in range(5):
            _ = sub_tensor.to(dst_dev)
        torch.cuda.synchronize()
        
        # 核心计时
        start_event.record()
        _ = sub_tensor.to(dst_dev)
        end_event.record()
        end_event.synchronize()
        
        t_us = start_event.elapsed_time(end_event)*1000
        
        data_list.append({
            "size_bytes": size,
            "time_us": t_us
        })

    # 4. 保存原始数据
    df = pd.DataFrame(data_list)
    output_file = "transfer_data_raw.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ 采集完成！原始数据已保存至: {output_file}")
    print(f"数据预览:\n{df.head()}")

if __name__ == "__main__":
    if torch.cuda.device_count() >= 2:
        collect_data()
    else:
        print("需要至少两张 GPU 才能运行此脚本。")

