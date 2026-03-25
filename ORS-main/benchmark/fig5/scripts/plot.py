#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_curve(data, batch_key):
    batch_data = data["data"][batch_key]
    seq_lens = sorted([int(k) for k in batch_data.keys()])
    throughput = [batch_data[str(x)]["throughput_req_s"] for x in seq_lens]
    return seq_lens, throughput

def plot_decode_throughput():
    # 1. 加载数据
    try:
        base_data = load_data("../results/baseline_throughput.json")
        ors_data = load_data("../results/throughput_results.json")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please run benchmarks first.")
        return

    # 2. 提取数据
    x_base, y_base = extract_curve(base_data, "batch_1")
    x_ors1, y_ors1 = extract_curve(ors_data, "batch_1")
    x_ors10, y_ors10 = extract_curve(ors_data, "batch_10")

    # 转换为等距类别轴（和你原代码一致）
    x_pos = np.arange(len(x_base))

    # 3. 开始绘图
    plt.figure(figsize=(8, 6))

    # --- Baseline ---
    plt.plot(x_pos, y_base, label='PyTorch Baseline (Concurrency=1)', 
             linewidth=2.5, marker='o', markersize=8,
             linestyle='-', color='#d62728')  # 红色

    # --- ORS batch=1 ---
    plt.plot(x_pos, y_ors1, label='ORS Framework (Concurrency=1)', 
             linewidth=2.5, marker='s', markersize=8,
             linestyle='-', color='#2ca02c')  # 绿色

    # --- ORS batch=10 ---
    plt.plot(x_pos, y_ors10, label='ORS Framework (Concurrency=10)', 
             linewidth=2.5, marker='^', markersize=8,
             linestyle='-', color='#1f77b4')  # 蓝色

    # 4. 坐标轴设置（保持你原来的风格）
    plt.xticks(x_pos, [str(x) for x in x_base], fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel('Context Length (Tokens in KV Cache)', fontsize=14, fontweight='bold')
    plt.ylabel('Throughput (Requests / Second)', fontsize=14, fontweight='bold')
    plt.title('Decode Throughput vs. Context Length (1000 requests)', fontsize=16)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=12)

    # Y轴范围（留空间）
    max_y = max(max(y_base), max(y_ors1), max(y_ors10))
    plt.ylim(0, max_y * 1.25)

    # 5. 保存
    os.makedirs("../results", exist_ok=True)
    plt.tight_layout()
    plt.savefig('../results/fig5_decode_throughput.pdf', dpi=300)
    plt.savefig('../results/fig5_decode_throughput.png', dpi=300)

    print("Plot saved to ../results/fig5_decode_throughput.(pdf/png)")


if __name__ == "__main__":
    plot_decode_throughput()