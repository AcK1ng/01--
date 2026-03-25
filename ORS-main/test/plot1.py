import matplotlib.pyplot as plt

def draw_chart():
    # 1. 数据准备
    frameworks = ['Baseline', 'ORS']
    latency = [18.5, 41.2]

    # 2. 创建画布
    plt.figure(figsize=(8, 6), dpi=300) # dpi=300 保证图片高清

    # 3. 绘制柱形图 (使用不同颜色区分)
    bars = plt.bar(frameworks, latency, color=['#5B9BD5', '#ED7D31'], width=0.5)

    # 4. 设置标签和标题
    # 注意：如果您的环境不支持中文显示，图表可能会出现乱码，
    # 为了通用性，这里主要使用英文标签，或者您可以手动指定支持中文的字体。
    plt.title('Framework Latency Comparison', fontsize=15, pad=20)
    plt.ylabel('Latency (s)', fontsize=12)
    plt.xlabel('Framework', fontsize=12)
    
    # 开启Y轴网格线，增加可读性
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.ylim(0, 50) # 设置Y轴范围，留出顶部空间

    # 5. 在柱子上方标注具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height}s',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 6. 保存图片 (bbox_inches='tight' 用于去除多余白边)
    filename = 'latency_comparison.png'
    plt.savefig(filename, bbox_inches='tight')
    
    print(f"图片已成功保存为: {filename}")

if __name__ == "__main__":
    draw_chart()
