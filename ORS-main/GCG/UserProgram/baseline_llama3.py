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
    for _ in range(500):
        token = generator.my_generate(kvcaches, 0,[0] * 1, 1, use_nvtx=False)
    sync_all_gpus()

    # 2. Benchmark Evaluation
    test_iters = 1000
    latencies = []
    
    print(f"[Baseline] Running benchmark for {test_iters} iterations...")
    for i in range(test_iters):
        # 使用 perf_counter 获取纳秒/微秒级高精度时间
        start_time = time.perf_counter() 
        
        token = generator.my_generate(kvcaches, 0, [0] * 1, 1, use_nvtx=False)
        sync_all_gpus()
        
        end_time = time.perf_counter()
        
        # 记录延迟 (转换为毫秒 ms)
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    # 3. 统计计算 (P50, P90, P99)
    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    p50_lat = np.percentile(latencies, 50)
    p90_lat = np.percentile(latencies, 90)
    p99_lat = np.percentile(latencies, 99)

    print("-" * 40)
    print(f"Baseline End-to-End Latency:")
    print(f"Mean : {mean_lat:.3f} ms")
    print(f"P50  : {p50_lat:.3f} ms")
    print(f"P90  : {p90_lat:.3f} ms")
    print(f"P99  : {p99_lat:.3f} ms")
    print("-" * 40)

    # 4. 导出数据供画图使用
    output_data = {
        "system": "PyTorch_Baseline",
        "metrics": {
            "mean_ms": mean_lat,
            "p50_ms": p50_lat,
            "p90_ms": p90_lat,
            "p99_ms": p99_lat
        },
        "raw_latencies_ms": latencies.tolist() 
    }

    # 确保输出目录存在
    os.makedirs("../results", exist_ok=True)
    out_path = "../results/baseline_latency.json"
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"[Baseline] Results saved to {out_path}")

if __name__ == "__main__":
    main()
