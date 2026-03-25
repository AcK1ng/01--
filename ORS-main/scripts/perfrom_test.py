import torch
import time
import matplotlib.pyplot as plt

def complex_fn(x, y):
    # step 1: 基础算子
    a = x + y
    b = x * y
    c = torch.matmul(a, b.T)  # 矩阵乘法

    # step 2: 激活与缩放
    d = torch.relu(c)
    e = torch.sigmoid(d) * 2.5

    # step 3: 归一化与拼接
    mean_val = torch.mean(e, dim=1, keepdim=True)
    std_val = torch.std(e, dim=1, keepdim=True)
    f = (e - mean_val) / (std_val + 1e-6)
    z = torch.cat([e, f], dim=1)

    return z

def benchmark_device(device, num_runs=1000, warmup=100):
    """测试指定设备的吞吐量"""
    print(f"正在测试 {device}...")
    
    # 创建测试数据
    batch_size = 128
    feature_size = 128
    x = torch.randn(batch_size, feature_size).to(device)
    y = torch.randn(batch_size, feature_size).to(device)
    
    # 预热
    for _ in range(warmup):
        _ = complex_fn(x, y)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 正式测试
    start_time = time.time()
    
    for _ in range(num_runs):
        _ = complex_fn(x, y)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = num_runs / total_time  # 请求/秒
    
    return throughput

def main():
    print("开始CPU和GPU性能对比测试...")
    
    # 检查GPU是否可用
    if torch.cuda.is_available():
        devices = [torch.device('cpu'), torch.device('cuda')]
        device_names = ['CPU', 'GPU']
    else:
        devices = [torch.device('cpu')]
        device_names = ['CPU']
        print("警告: 未检测到GPU，仅测试CPU性能")
    
    # 测试参数
    num_runs = 1000
    warmup = 100
    
    # 运行测试
    throughputs = []
    for device in devices:
        throughput = benchmark_device(device, num_runs, warmup)
        throughputs.append(throughput)
        print(f"{device} 吞吐量: {throughput:.2f} req/s")

if __name__ == "__main__":
    main()
