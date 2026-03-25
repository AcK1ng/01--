import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. 读取数据 ===
data = pd.read_csv('../raw/data.csv', sep=r'\s+')

# 提取横坐标
infer_counts = data['infer_count']
x = np.arange(len(infer_counts))

# 时间分解列（排除 total_infer_time）
time_columns = ['copy_graph_input_time', 'prepare_input_time', 'execute_op_time', 'extract_output_time', 'notify_op_time', 'schedule_time']

# === 2. 计算百分比 ===
time_sum = data[time_columns].sum(axis=1)
percent_data = data[time_columns].div(time_sum, axis=0) * 100  # 每列换算成百分比

# === 3. 颜色与样式设置（论文风格） ===
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']

# === 4. 绘图区域设置 ===
plt.figure(figsize=(10, 8))
bottom = np.zeros(len(data))

# === 5. 绘制堆叠柱状图（百分比） ===
for i, col in enumerate(time_columns):
    print(percent_data[col])
    plt.bar(x, percent_data[col], bottom=bottom, color=colors[i],
            label=col.replace('_', ' ').title(), width=0.6,
            edgecolor='white', linewidth=0.7)

    # 添加百分比标注
    for j in range(len(data)):
        height = percent_data[col].iloc[j]
        if height > 5:  # 只在占比>5%时显示
            plt.text(x[j], bottom[j] + height / 2,
                     f'{height:.1f}%', ha='center', va='center',
                     fontsize=9, color='white', fontweight='bold')
    bottom += percent_data[col]

# === 6. 坐标轴与标题样式 ===
#plt.title('Time Breakdown Percentage per Inference Count', fontsize=14, fontweight='bold')
plt.xlabel('Infer Count', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(x, infer_counts, fontsize=11)
plt.yticks(np.arange(0, 101, 10), fontsize=11)
plt.ylim(0, 100)
plt.title('GPU Time Percentage', fontsize=14, fontweight='bold')

# 优化网格与边框
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.box(False)

# === 7. 图例设置 ===
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=10, frameon=False)

# === 8. 布局与保存 ===
plt.tight_layout()
plt.savefig('time_breakdown_percentage.png', dpi=300, bbox_inches='tight')
plt.show()
