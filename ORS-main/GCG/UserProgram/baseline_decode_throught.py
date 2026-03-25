#!/usr/bin/env python3
import os
import time
import json
import torch

from llama3_8b_instruct_infer import Llama


def sync_all_gpus():
    if torch.cuda.device_count() > 0:
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device=torch.device(f'cuda:{device_id}'))


def run_throughput_benchmark(generator, kvcaches, seq_len, test_iters):
    print(f"\n=== Running Throughput Benchmark | seq_len={seq_len} ===")

    # ===== 构造 decode 输入（与 C++ 对齐）=====
    local_hit = seq_len
    prompt_tokens = [0] * (seq_len + 1)   # 保持你原来的逻辑（稳定）
    max_gen_len = 1

    # ===== Warmup（对齐 C++）=====
    for _ in range(10):
        generator.my_generate(
            kvcaches,
            local_hit=local_hit,
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            use_nvtx=False
        )
    sync_all_gpus()

    # ===== Benchmark（串行执行）=====
    start_time = time.perf_counter()

    for _ in range(test_iters):
        generator.my_generate(
            kvcaches,
            local_hit=local_hit,
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            use_nvtx=False
        )
        sync_all_gpus()  # ⚠️ 必须有，否则吞吐虚高

    end_time = time.perf_counter()

    total_time = end_time - start_time
    throughput = test_iters / total_time

    print(f"Total Time: {total_time:.4f}s")
    print(f"Throughput: {throughput:.2f} req/s")

    return throughput


def save_results(filepath, results):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filepath}")


def main():
    ckpt_dir = "llama3_8b_instruct_infer/"
    tokenizer_path = "llama3_8b_instruct_infer/tokenizer.model"
    max_kv_capacity = 131072 + 1

    print("[Baseline] Loading model...")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_kv_capacity,
        temperature=0.6,
        top_p=0.9
    )

    # 与 C++ 对齐
    context_lengths = [1, 4, 16, 32, 64, 128, 512, 1024, 2048, 4096]
    test_iters = 1000

    print("[Baseline] Pre-allocating KV cache...")
    kvcaches = generator.generate_kvcaches(max_kv_capacity)

    results = {
        "system": "PyTorch_Baseline",
        "benchmark": "Decode_Throughput",
        "data": {}
    }

    # ⚠️ 注意：这里仍然保留 batch_1 / batch_10 结构（对齐 C++ JSON）
    # 但串行版本本质等价于 batch_1
    for batch_size in [1, 10]:
        batch_key = f"batch_{batch_size}"
        results["data"][batch_key] = {}

        for seq_len in context_lengths:
            throughput = run_throughput_benchmark(
                generator,
                kvcaches,
                seq_len,
                test_iters
            )

            results["data"][batch_key][str(seq_len)] = {
                "throughput_req_s": round(throughput, 2)
            }

    save_results("../results/baseline_throughput.json", results)


if __name__ == "__main__":
    main()