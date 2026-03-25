#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_multi_thread_mean_latency():
    # 1. 加载数据
    try:
        json_data = load_data("../results/ors_thread_decode_latency.json")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the JSON file exists.")
        return

    data_dict = json_data["data"]
    
    # 2. 提取 X 轴 (Context Lengths) 并确保按数字大小排序
    first_thread_key = list(data_dict.keys())[0]
    seq_lens_str = list(data_dict[first_thread_key].keys())
    seq_lens = sorted([int(x) for x in seq_lens_str])
    x_pos = np.arange(len(seq_lens))

    # 3. 准备画布
    plt.figure(figsize=(10, 7))

    # 定义颜色和标记字典，用于区分不同线程
    style_map = {
        0: {'color': '#1f77b4', 'marker': 'o', 'label': '1 Thread'}, # 蓝色
        1: {'color': '#d62728', 'marker': 's', 'label': '4 Threads'},# 红色
        2: {'color': '#2ca02c', 'marker': '^', 'label': '8 Threads'}, # 绿色
        3: {'color': '#9467bd', 'marker': 'D', 'label': '16 Threads'} # 紫色
    }

    max_y_value = 0 # 用于动态计算 Y 轴上限

    # 4. 遍历每个线程配置，只绘制 Mean 延迟
    for idx, (thread_name, ctx_data) in enumerate(data_dict.items()):
        
        # 只提取 Mean
        mean_vals = [ctx_data[str(x)]["mean_ms"] for x in seq_lens]
        
        # 记录最大值
        max_y_value = max(max_y_value, max(mean_vals))

        style = style_map[idx % len(style_map)]
        c = style['color']
        m = style['marker']
        
        # 格式化图例标签
        label_base = thread_name.replace("_", " ").title()

        # --- 只绘制 Mean 延迟 (使用实线和清晰的 Marker) ---
        plt.plot(x_pos, mean_vals, 
                 label=label_base, 
                 linewidth=2.5, marker=m, markersize=8, linestyle='-')
                 
    # 5. 图表修饰
    plt.xticks(x_pos, [str(x) for x in seq_lens], fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.xlabel('Context Length (Tokens in KV Cache)', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Decode Step Latency (ms)', fontsize=14, fontweight='bold')
    
    system_name = json_data.get("system", "ORS Framework")
    plt.title(f'{system_name}: Mean Latency vs. Context Length by Thread Count', fontsize=16)
    
    plt.grid(True, which='major', linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=12)
    
    # Y 轴从 0 开始，顶部留出 20% 空间
    plt.ylim(0, max_y_value * 1.25)

    # 6. 保存图表
    os.makedirs("../results", exist_ok=True)
    plt.tight_layout()
    # 使用新文件名，避免覆盖之前的图
    plt.savefig('../results/fig_mean_latency_by_thread.pdf', dpi=300)
    plt.savefig('../results/fig_mean_latency_by_thread.png', dpi=300)
    print("Plot successfully saved to ../results/fig_mean_latency_by_thread.pdf and .png")

if __name__ == "__main__":
    plot_multi_thread_mean_latency()