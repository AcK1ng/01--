#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_context_length():
    try:
        base_data = load_data("../results/baseline_context_latency.json")
        ors_data = load_data("../results/ors_context_latency.json")
    except FileNotFoundError as e:
        print(f"Error: {e}. Run the benchmarks first.")
        return

    # 提取 X 轴 (Context Lengths) 并确保它是按数字大小排序的
    seq_lens_str = list(base_data["data"].keys())
    seq_lens = sorted([int(x) for x in seq_lens_str])
    
    # 提取 Y 轴数据 (以 P99 和 Mean 为例)
    base_mean = [base_data["data"][str(x)]["mean_ms"] for x in seq_lens]
    base_p99 = [base_data["data"][str(x)]["p99_ms"] for x in seq_lens]
    
    ors_mean = [ors_data["data"][str(x)]["mean_ms"] for x in seq_lens]
    ors_p99 = [ors_data["data"][str(x)]["p99_ms"] for x in seq_lens]

    plt.figure(figsize=(9, 6))

    # 设置 X 轴的刻度为文字，防止 128, 256 等被 matplotlib 压成非等距
    x_pos = np.arange(len(seq_lens))

    # --- 绘制 P99 延迟 (用粗线 + 明显标记) ---
    plt.plot(x_pos, base_p99, label='PyTorch Baseline (P99)', 
             linewidth=2.5, marker='o', markersize=8, linestyle='-', color='#d62728')  # 红色实线
             
    plt.plot(x_pos, ors_p99, label='ORS Framework (P99)', 
             linewidth=2.5, marker='s', markersize=8, linestyle='-', color='#2ca02c')   # 绿色实线

    # --- 绘制 Mean 延迟 (用细虚线做对比参考) ---
    plt.plot(x_pos, base_mean, label='PyTorch Baseline (Mean)', 
             linewidth=1.5, marker='o', markersize=5, linestyle='--', color='#d62728', alpha=0.6)
             
    plt.plot(x_pos, ors_mean, label='ORS Framework (Mean)', 
             linewidth=1.5, marker='s', markersize=5, linestyle='--', color='#2ca02c', alpha=0.6)

    # 图表修饰
    plt.xticks(x_pos,[str(x) for x in seq_lens], fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.xlabel('Context Length (Tokens)', fontsize=14, fontweight='bold')
    plt.ylabel('End-to-End Latency (ms)', fontsize=14, fontweight='bold')
    plt.title('Latency vs. Context Length (1000 requests)', fontsize=16)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=12)
    
    # 学术论文常对坐标轴进行小范围的余量设置，让图更好看
    plt.ylim(0, max(base_p99) * 1.15)

    plt.tight_layout()
    plt.savefig('../results/fig3_context_length.pdf', dpi=300)
    plt.savefig('../results/fig3_context_length.png', dpi=300)
    print("Plot successfully saved to ../results/fig3_context_length.pdf/.png")

if __name__ == "__main__":
    plot_context_length()