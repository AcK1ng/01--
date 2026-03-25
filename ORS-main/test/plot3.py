import matplotlib.pyplot as plt

# 1. 数据准备 (已添加 ORS-Libtorch)
# 使用 \n 换行，防止标签太长挤在一起
frameworks = ['Baseline', 'ORS', 'ORS\n(Libtorch)'] 
latency = [18.5, 41.2, 12.5]

# 2. 创建画布
plt.figure(figsize=(9, 6), dpi=300) # 稍微加宽一点画布，并设置高清DPI

# 3. 绘制柱形图
# 颜色依次为：蓝色(Baseline), 红色(慢), 绿色(快/Libtorch)
colors = ['#4c72b0', '#ED7D31', '#55a868']
bars = plt.bar(frameworks, latency, color=colors, width=0.5)

# 4. 设置标题和标签
# 注意：如果您的系统中没有默认的中文字体配置，中文可能会显示为方框。
# 如果出现乱码，建议临时将中文改为英文，或者手动设置 font.sans-serif
plt.title('Framework Latency Comparison', fontsize=15, pad=20)
plt.ylabel('Latency (s)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 5. 在柱子上方添加具体数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height}s',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# 6. 设置Y轴上限
plt.ylim(0, 50)

# 7. 保存图片
filename = 'latency_comparison_v3.png'
plt.savefig(filename, bbox_inches='tight')

print(f"图片已生成并保存为: {filename}")
