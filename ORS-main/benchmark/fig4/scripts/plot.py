#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_decode_context():
    # 1. 尝试加载数据
    try:
        base_data = load_data("../results/baseline_decode_latency.json")
        ors_data = load_data("../results/ors_decode_latency.json")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please run both baseline_decode.py and the ORS C++ binary first.")
        return

    # 2. 提取 X 轴 (Context Lengths) 并确保按数字大小排序
    seq_lens_str = list(base_data["data"].keys())
    seq_lens = sorted([int(x) for x in seq_lens_str])
    
    # 3. 提取 Y 轴数据
    base_mean = [base_data["data"][str(x)]["mean_ms"] for x in seq_lens]
    base_p99 = [base_data["data"][str(x)]["p99_ms"] for x in seq_lens]
    
    ors_mean = [ors_data["data"][str(x)]["mean_ms"] for x in seq_lens]
    ors_p99 = [ors_data["data"][str(x)]["p99_ms"] for x in seq_lens]

    # 4. 开始绘图 (宽比例适合学术论文单栏)
    plt.figure(figsize=(8, 6))

    # 设置 X 轴的刻度位置 (等距类别轴)
    x_pos = np.arange(len(seq_lens))

    # --- 绘制 P99 延迟 (使用实线和较大的 Marker) ---
    plt.plot(x_pos, base_p99, label='PyTorch Baseline (P99)', 
             linewidth=2.5, marker='o', markersize=8, linestyle='-', color='#d62728')  # 红色
             
    plt.plot(x_pos, ors_p99, label='ORS Framework (P99)', 
             linewidth=2.5, marker='s', markersize=8, linestyle='-', color='#2ca02c')   # 绿色

    # --- 绘制 Mean 延迟 (使用细虚线做辅助对比) ---
    plt.plot(x_pos, base_mean, label='PyTorch Baseline (Mean)', 
             linewidth=1.5, marker='o', markersize=5, linestyle='--', color='#d62728', alpha=0.6)
             
    plt.plot(x_pos, ors_mean, label='ORS Framework (Mean)', 
             linewidth=1.5, marker='s', markersize=5, linestyle='--', color='#2ca02c', alpha=0.6)

    # 5. 图表修饰 (学术标准)
    # 将 X 轴的刻度替换回真实的 Context Length 数值
    plt.xticks(x_pos, [str(x) for x in seq_lens], fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.xlabel('Context Length (Tokens in KV Cache)', fontsize=14, fontweight='bold')
    plt.ylabel('Single Decode Step Latency (ms)', fontsize=14, fontweight='bold')
    plt.title('Decode Latency vs. Context Length (1000 requests)', fontsize=16)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=12)
    
    # Y 轴从 0 开始，顶部留出 15% 的空间放图例
    max_y = max(max(base_p99), max(ors_p99))
    plt.ylim(0, max_y * 1.25)

    # 6. 保存图表
    os.makedirs("../results", exist_ok=True)
    plt.tight_layout()
    plt.savefig('../results/fig4_decode_context.pdf', dpi=300)
    plt.savefig('../results/fig4_decode_context.png', dpi=300)
    print("Plot successfully saved to benchmark/fig4_decode_context/results/fig4_decode_context.pdf and .png")

if __name__ == "__main__":
    plot_decode_context()