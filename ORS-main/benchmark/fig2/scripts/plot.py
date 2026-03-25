#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_cdf():
    try:
        base_data = load_data("../results/baseline_macro_latency.json")
        ors_data = load_data("../results/ors_macro_latency.json")
    except FileNotFoundError as e:
        print(f"Error: {e}. Run the benchmarks first.")
        return

    # 因为数据是连续 1000 个请求的总耗时，数值很大(毫秒级)，我们除以 1000 转换为 秒(s)
    base_lats_sec = np.sort(base_data["raw_latencies_ms"]) / 1000.0
    ors_lats_sec = np.sort(ors_data["raw_latencies_ms"]) / 1000.0

    base_y = np.arange(1, len(base_lats_sec) + 1) / len(base_lats_sec)
    ors_y = np.arange(1, len(ors_lats_sec) + 1) / len(ors_lats_sec)

    plt.figure(figsize=(8, 6))
    
    # 绘制曲线
    plt.plot(base_lats_sec, base_y, label='PyTorch Baseline', 
             linewidth=2.5, linestyle='--', color='#ff7f0e')
             
    plt.plot(ors_lats_sec, ors_y, label='ORS Framework (Ours)', 
             linewidth=2.5, linestyle='-', color='#1f77b4')

    # 标出 P99
    base_p99_sec = base_data["metrics"]["p99_ms"] / 1000.0
    ors_p99_sec = ors_data["metrics"]["p99_ms"] / 1000.0
    
    plt.axhline(y=0.99, color='gray', linestyle=':', alpha=0.7)
    
    plt.scatter([base_p99_sec],[0.99], color='#ff7f0e', zorder=5, s=50)
    plt.scatter([ors_p99_sec],[0.99], color='#1f77b4', zorder=5, s=50)

    # 标注具体数值
    plt.text(ors_p99_sec, 0.92, f"P99: {ors_p99_sec:.2f}s", color='#1f77b4', fontweight='bold')
    plt.text(base_p99_sec*0.9, 0.92, f"P99: {base_p99_sec:.2f}s", color='#ff7f0e', fontweight='bold')

    # 图表修饰
    plt.xlabel('Total Latency for 1000 Sequential Requests (Seconds)', fontsize=14, fontweight='bold')
    plt.ylabel('CDF', fontsize=14, fontweight='bold')
    plt.title('Macro-Stability: System Jitter over Heavy Workloads', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('../results/fig2_macro_latency_cdf.pdf', dpi=300)
    plt.savefig('../results/fig2_macro_latency_cdf.png', dpi=300)
    print("Plot successfully saved to ../results/")

if __name__ == "__main__":
    plot_cdf()