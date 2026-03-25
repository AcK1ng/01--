import pandas as pd
import numpy as np
import json
from scipy.stats import linregress
import os

def train_model(csv_path="transfer_data_raw.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: 找不到文件 {csv_path}，请先运行 1_collect_data.py")
        return

    print(f"=== 开始训练模型 (输入: {csv_path}) ===")
    df = pd.read_csv(csv_path)
    
    # 按大小排序
    df = df.sort_values('size_bytes').reset_index(drop=True)
    
    sizes = df['size_bytes'].values
    times = df['time_us'].values
    n = len(df)

    best_mse = float('inf')
    best_params = None

    # === 暴力搜索最佳分割点 (Threshold) ===
    # 我们假设拐点一定在第 10 个点到倒数第 10 个点之间
    # 这样保证两段都有足够的数据来拟合
    search_range = range(10, n - 10)
    
    for i in search_range:
        # 1. 切分数据
        # Part 1 (小数据): 假设是 Latency Bound，时间近似常数
        s1 = sizes[:i]
        t1 = times[:i]
        
        # Part 2 (大数据): 假设是 Bandwidth Bound，线性增长
        s2 = sizes[i:]
        t2 = times[i:]
        
        # 2. 拟合左边 (取均值作为 Base Latency)
        latency_val = np.mean(t1)
        pred1 = np.full_like(t1, latency_val)
        
        # 3. 拟合右边 (线性回归)
        slope, intercept, r_val, _, _ = linregress(s2, t2)
        # 约束：斜率必须为正 (带宽不能是负的)
        if slope <= 0: continue
            
        pred2 = intercept + slope * s2
        
        # 4. 计算总均方误差 (MSE)
        error = np.sum((t1 - pred1)**2) + np.sum((t2 - pred2)**2)
        
        # 5. 更新最佳结果
        if error < best_mse:
            best_mse = error
            # 记录参数
            bandwidth_gbps = (1 / slope) / 1e9 * 1000000 # us/Byte -> GB/s
            
            best_params = {
                "threshold_bytes": int(sizes[i]),
                "threshold_kb": round(sizes[i] / 1024, 2),
                "base_latency_us": float(latency_val),
                "bandwidth_gbps": float(bandwidth_gbps),
                "linear_slope": float(slope),
                "linear_intercept": float(intercept),
                "r_squared": float(r_val**2)
            }

    # === 输出与保存 ===
    if best_params:
        print("\n模型训练成功！最佳参数如下：")
        print(f"  - 阈值 (Threshold)  : {best_params['threshold_kb']} KB")
        print(f"  - 基础延迟 (Latency): {best_params['base_latency_us']:.4f} us")
        print(f"  - 传输带宽 (BW)     : {best_params['bandwidth_gbps']:.2f} GB/s")
        print(f"  - 线性拟合 R²       : {best_params['r_squared']:.4f}")

        output_file = "transfer_profile.json"
        with open(output_file, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"\n参数已保存至: {output_file}")
    else:
        print("❌ 训练失败，无法找到合适的拟合模型。请检查原始数据是否异常。")

if __name__ == "__main__":
    train_model()

