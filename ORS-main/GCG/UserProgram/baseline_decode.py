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
    max_kv_capacity: int = 131072 + 1  # 128k tokens + 1 token for current decoding step

    print("[Baseline] Loading model...")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_kv_capacity,
        temperature=0.6,
        top_p=0.9
    )

    context_lengths = [1, 4, 16, 32, 64, 128, 512, 1024, 2048, 4096, 8192, 16384]
    
    test_iters = 1000
    
    results = {"system": "PyTorch_Baseline_Decode", "data": {}}

    print(f"[Baseline] Running DECODE context length benchmark via my_generate...")

    # 预分配最大容量的 KV Caches
    kvcaches = generator.generate_kvcaches(max_kv_capacity)

    for seq_len in context_lengths:
        print(f"\n--- Testing Decode at Context Length: {seq_len} ---")
        
        # 巧妙构造参数，强制 my_generate 进入 "单步 Decode" 模式
        # 1. 历史缓存长度为 seq_len
        local_hit = seq_len 
        # 2. 加上当前要 Decode 的 1 个 token，总 prompt 长度为 seq_len + 1
        prompt_tokens = [0] * (seq_len + 1)
        # 3. 限制只生成 1 步
        max_gen_len = 1 
        
        # 1. Warmup
        for _ in range(5):
            _ = generator.my_generate(
                kvcaches, 
                local_hit=local_hit, 
                prompt_tokens=prompt_tokens, 
                max_gen_len=max_gen_len, 
                use_nvtx=False
            )
        sync_all_gpus()

        # 2. Benchmark
        latencies = []
        for _ in range(test_iters):
            start_time = time.perf_counter() 
            
            # 直接调用完整的 Python generate 流程
            _ = generator.my_generate(
                kvcaches, 
                local_hit=local_hit, 
                prompt_tokens=prompt_tokens, 
                max_gen_len=max_gen_len, 
                use_nvtx=False
            )
            sync_all_gpus()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

        # 3. 统计计算
        latencies = np.array(latencies)
        mean_lat = np.mean(latencies)
        p99_lat = np.percentile(latencies, 99)

        print(f"Decode SeqLen {seq_len} | Mean: {mean_lat:.3f}ms | P99: {p99_lat:.3f}ms")

        results["data"][str(seq_len)] = {
            "mean_ms": mean_lat, "p50_ms": np.percentile(latencies, 50),
            "p90_ms": np.percentile(latencies, 90), "p99_ms": p99_lat
        }

    os.makedirs("../results", exist_ok=True)
    with open("../results/baseline_decode_latency.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()