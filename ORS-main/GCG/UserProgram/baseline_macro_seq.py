#!/usr/bin/env python3
import os
import time
import json
import torch
import numpy as np
from llama3_8b_instruct_infer import Dialog, Llama

def sync_all_gpus():
    if torch.cuda.device_count() > 0:
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device=torch.device(f'cuda:{device_id}'))

def main():
    ckpt_dir: str = "llama3_8b_instruct_infer/"
    tokenizer_path: str = "llama3_8b_instruct_infer/tokenizer.model"
    temperature: float = 0.6
    top_p: float = 0.9
    max_seq_len: int = 16 * 1024

    print("[Baseline] Loading model...")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        temperature=temperature,
        top_p=top_p
    )

    kvcaches = generator.generate_kvcaches(128)
    
    # 1. Warmup
    print("[Baseline] Warming up for 5 iterations...")
    for _ in range(5):
        _ = generator.my_generate(kvcaches, 0, [0] * 1, 1, use_nvtx=False)
    sync_all_gpus()

    # 2. Benchmark Evaluation
    num_samples = 50       # 外层循环：收集 1000 个数据点用于画 CDF
    reqs_per_sample = 1000   # 内层循环：每个数据点是 1000 个请求的总和
    latencies =[]
    
    print(f"[Baseline] Running macro benchmark ({num_samples} samples x {reqs_per_sample} requests)...")
    
    for i in range(num_samples):
        # 记录 1000 个请求的起始时间
        start_time = time.perf_counter() 
        
        # 串行执行 1000 个请求
        for j in range(reqs_per_sample):
            _ = generator.my_generate(kvcaches, 0, [0] * 1, 1, use_nvtx=False)
            sync_all_gpus()  # 严格串行：必须等上一个做完再发下一个
            
        # 记录 1000 个请求结束的时间
        end_time = time.perf_counter()
        
        # 记录这个 Workload 的总延迟 (ms)
        workload_latency_ms = (end_time - start_time) * 1000
        latencies.append(workload_latency_ms)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_samples} samples done. Last sample latency: {workload_latency_ms:.2f} ms")

    # 3. 统计计算 (P50, P90, P99)
    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    p50_lat = np.percentile(latencies, 50)
    p90_lat = np.percentile(latencies, 90)
    p99_lat = np.percentile(latencies, 99)

    print("-" * 50)
    print(f"Baseline Macro-Sequential Latency (1000 reqs/sample):")
    print(f"Mean : {mean_lat:.3f} ms")
    print(f"P50  : {p50_lat:.3f} ms")
    print(f"P90  : {p90_lat:.3f} ms")
    print(f"P99  : {p99_lat:.3f} ms")
    print("-" * 50)

    # 4. 导出数据
    output_data = {
        "system": "PyTorch_Baseline",
        "workload": f"{reqs_per_sample}_sequential_requests",
        "metrics": {
            "mean_ms": mean_lat, "p50_ms": p50_lat, "p90_ms": p90_lat, "p99_ms": p99_lat
        },
        "raw_latencies_ms": latencies.tolist() 
    }

    os.makedirs("../results", exist_ok=True)
    with open("../results/baseline_macro_latency.json", "w") as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    main()
