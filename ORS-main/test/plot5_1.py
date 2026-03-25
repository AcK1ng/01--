import matplotlib.pyplot as plt

# 1. 数据准备 (已添加 ORS-Libtorch 和 ORS-optimization)
# 使用 \n 换行，防止标签太长挤在一起
frameworks = ['Baseline', 'Libtorch', 'ORS', 'ORS\n(single_thread\nsingle_stream)', 'ORS\n(multi_thread\nsingle_stream)', 'ORS\n(multi_thread\nmulti_stream)'] 
latency = [18.5, 12.5, 41.2, 16.1, 15.9, 31.8]

# 2. 创建画布
plt.figure(figsize=(10, 6), dpi=300)  # 增加宽度以容纳更多柱子

# 3. 绘制柱形图
# 颜色：蓝色(Baseline), 橙色(慢), 紫色(优化), 绿色(最快/Libtorch)
colors = ['#4c72b0', '#ED7D31', '#9467bd', '#55a868', '#72BCD4', '#F2C80F']
bars = plt.bar(frameworks, latency, color=colors, width=0.5)

# 4. 设置标题和标签
plt.title('Framework Latency Comparison', fontsize=16, pad=20, fontweight='bold')
plt.ylabel('Latency (s)', fontsize=13)
plt.xlabel('Framework', fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 5. 在柱子上方添加具体数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height}s',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 6. 添加相对于Baseline的百分比变化标注
baseline = latency[0]
for i, bar in enumerate(bars):
    if i > 0:  # 跳过Baseline自身
        height = bar.get_height()
        change = ((height - baseline) / baseline) * 100
        change_text = f"{'+' if change > 0 else ''}{change:.0f}%"
        
        # 在柱子中间位置显示百分比
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                 change_text,
                 ha='center', va='center', fontsize=10,
                 color='white' if height > baseline else 'white',
                 fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.5))

# 7. 添加基准参考线
plt.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, 
            linewidth=1.5, label=f'Baseline ({baseline}s)')

# 8. 设置Y轴上限（稍微提高以容纳新增的柱子和标签）
max_latency = max(latency)
plt.ylim(0, max_latency * 1.4)

# 9. 添加图例说明各颜色含义
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors[0], label=f'Baseline: {latency[0]}s'),
    Patch(facecolor=colors[1], label=f'ORS Original: {latency[1]}s (+123%)'),
    Patch(facecolor=colors[2], label=f'ORS Optimization: {latency[2]}s (+26%)'),
    Patch(facecolor=colors[3], label=f'ORS Libtorch: {latency[3]}s (-32%)')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

# 10. 调整布局
plt.tight_layout()

# 11. 保存图片
filename = 'latency_comparison_with_optimization.png'
plt.savefig(filename, bbox_inches='tight', dpi=300)

print(f"图片已生成并保存为: {filename}")
