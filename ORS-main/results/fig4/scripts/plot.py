import pandas as pd
import matplotlib.pyplot as plt

# === 1. 读取数据 ===
data = pd.read_csv('../raw/data.csv', sep=r'\s+')
# 文件结构: 第一行是列名 (pytorch, ors)，第二行是数值
# 所以我们取出第一行列名、第二行数值
methods = list(data.columns)
values = data.iloc[0].astype(float).tolist()

# === 2. 绘图设置 ===
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(6, 5))

# === 3. 绘制柱状图 ===
bars = plt.bar(methods, values, color=['#4C72B0', '#55A868'], width=0.6)

# === 4. 添加数值标注 ===
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + max(values)*0.02,
             f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# === 5. 坐标轴与标题 ===
plt.ylabel('Throughput (samples/sec)', fontsize=12)
plt.xlabel('Method', fontsize=12)
plt.title('CPU Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim(0, max(values) * 1.25)
plt.tight_layout()

# === 6. 保存与显示 ===
plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

