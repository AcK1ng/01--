#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_cdf():
    # 1. 加载数据
    try:
        base_data = load_data("../results/baseline_latency.json")
        ors_data = load_data("../results/ors_latency.json")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please run both baseline.py and the ORS C++ binary first.")
        return

    base_lats = np.sort(base_data["raw_latencies_ms"])
    ors_lats = np.sort(ors_data["raw_latencies_ms"])

    # 计算 CDF 的 Y 轴百分比 (0 到 1)
    base_y = np.arange(1, len(base_lats) + 1) / len(base_lats)
    ors_y = np.arange(1, len(ors_lats) + 1) / len(ors_lats)

    # 2. 绘制 CDF 图形
    plt.figure(figsize=(8, 6))
    
    # 画 Baseline (橙色虚线)
    plt.plot(base_lats, base_y, label='PyTorch Baseline', 
             linewidth=2.5, linestyle='--', color='#ff7f0e') 
             
    # 画 ORS (蓝色实线)
    plt.plot(ors_lats, ors_y, label='ORS Framework (Ours)', 
             linewidth=2.5, linestyle='-', color='#1f77b4')  

    # --- 标出 P99 和 P90 的位置 ---
    base_p99 = base_data["metrics"]["p99_ms"]
    ors_p99 = ors_data["metrics"]["p99_ms"]
    
    # P99 辅助线
    plt.axhline(y=0.99, color='gray', linestyle=':', alpha=0.7)
    plt.text(min(base_p99, ors_p99)*0.9, 0.995, '99th Percentile', color='gray', fontsize=10)

    # 用点把 P99 标出来
    plt.scatter([base_p99],[0.99], color='#ff7f0e', zorder=5, s=50)
    plt.scatter([ors_p99], [0.99], color='#1f77b4', zorder=5, s=50)

    # 图表修饰 (学术论文标准样式)
    plt.xlabel('End-to-End Latency (ms)', fontsize=14, fontweight='bold')
    plt.ylabel('CDF (Cumulative Probability)', fontsize=14, fontweight='bold')
    plt.title('Single Request Latency Distribution', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    plt.ylim(0, 1.05)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 3. 保存图表
    plt.tight_layout()
    plt.savefig('../results/fig1_latency_cdf.pdf', dpi=300)
    plt.savefig('../results/fig1_latency_cdf.png', dpi=300)
    print("Plot successfully saved to benchmark/fig1_latency/results/fig1_latency_cdf.pdf and .png")

if __name__ == "__main__":
    plot_cdf()
