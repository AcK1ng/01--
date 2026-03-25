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
    
    # 物理显存分配的最大长度设为 2048
    max_kv_capacity: int = 2048

    print("[Baseline] Loading model...")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_kv_capacity,
        temperature=temperature,
        top_p=top_p
    )

    # 添加长度 1，模拟极短文本或 Decode 阶段的单 Token 负载
    context_lengths =[128, 256, 512, 1024]
    test_iters = 1000
    
    results = {
        "system": "PyTorch_Baseline",
        "data": {}
    }

    print(f"[Baseline] Running context length benchmark...")

    # 统一分配 KV Cache（排除显存分配的干扰）
    kvcaches = generator.generate_kvcaches(max_kv_capacity)

    for seq_len in context_lengths:
        print(f"\n--- Testing Context Length: {seq_len} ---")
        
        # 动态生成对应长度的输入 tokens
        # 这里用 [0]*seq_len 模拟长度为 seq_len 的 prompt
        input_tokens = [0] * seq_len
        
        # 1. Warmup
        for _ in range(5):
            _ = generator.my_generate(kvcaches, 0, input_tokens, 1, use_nvtx=False)
        sync_all_gpus()

        # 2. Benchmark
        latencies =[]
        for _ in range(test_iters):
            start_time = time.perf_counter() 
            _ = generator.my_generate(kvcaches, 0, input_tokens, 1, use_nvtx=False)
            sync_all_gpus()
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

        # 3. 统计计算
        latencies = np.array(latencies)
        mean_lat = np.mean(latencies)
        p50_lat = np.percentile(latencies, 50)
        p90_lat = np.percentile(latencies, 90)
        p99_lat = np.percentile(latencies, 99)

        print(f"SeqLen {seq_len} | Mean: {mean_lat:.2f}ms | P99: {p99_lat:.2f}ms")

        results["data"][str(seq_len)] = {
            "mean_ms": mean_lat,
            "p50_ms": p50_lat,
            "p90_ms": p90_lat,
            "p99_ms": p99_lat
        }

    # 4. 导出汇总数据
    os.makedirs("../results", exist_ok=True)
    with open("../results/baseline_context_latency.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n[Baseline] Results saved to ../results/baseline_context_latency.json")

if __name__ == "__main__":
    main()
