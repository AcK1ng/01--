import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV
df = pd.read_csv("../raw/data.csv", sep=r'\s+')

# 先查看实际列名
print("实际列名:", df.columns.tolist())

# 根据实际列名修改下面的变量名
# 例如，如果列名是 "infer_count", "pytorch", "ors" 就保持不变
# 如果列名不同，需要相应修改

infer_count = df['infer_count'].tolist()  # 如果列名不同，修改这里
pytorch = df["pytorch"].tolist()          # 如果列名不同，修改这里  
ors = df["ors"].tolist()                  # 如果列名不同，修改这里

# 论文风格图
plt.figure(figsize=(8, 5))
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

# 设置对数坐标轴
# plt.xscale("log")

# 绘制曲线（不同 marker）
plt.plot(
    infer_count, pytorch, marker='s', markersize=8, linewidth=2, label="PyTorch"
)
plt.plot(
    infer_count, ors, marker='^', markersize=8, linewidth=2, label="ORS"
)

# 标签与标题
plt.xlabel("task count", fontsize=13)
plt.ylabel("Throughput", fontsize=13)
plt.title("GPU Performance Comparison", fontsize=14)

plt.legend(fontsize=12)
plt.tight_layout()

# 保存为 PNG
plt.savefig("gpu_performance_comparison.png", dpi=300)
