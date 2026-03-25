import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取数据
data = pd.read_csv('../raw/data.csv', sep=r'\s+')
methods = list(data.columns)
values = data.iloc[0].astype(float)

# # 2. 以第一列为基准归一化
# base = values.iloc[0]                  # 使用 iloc 避免 FutureWarning
# normalized_values = (values / base).tolist()

# 3. 绘图
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(7,5))
colors = ['#4C72B0', '#55A868', '#C44E52']
bars = plt.bar(methods, values, color=colors[:len(values)], width=0.6)

# 4. 添加数值标注
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + max(values)*0.02,
             f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 5. 坐标轴与标题
plt.ylabel('Normalized to First Column', fontsize=12)
plt.xlabel('Method', fontsize=12)
plt.title('Infer Latency', fontsize=14, fontweight='bold')
plt.ylim(0, max(values) * 1.25)
plt.tight_layout()

# 6. 保存与显示
plt.savefig('method_comparison_normalized_to_first.png', dpi=300, bbox_inches='tight')
plt.show()
