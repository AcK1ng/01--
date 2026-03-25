import torch
import torch.nn as nn
import time

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(1, 20, 4),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5),
            nn.MaxPool2d(3),
            nn.Flatten(),
            nn.Linear(40 * 3 * 3, 150),
            nn.Linear(150, 10),
        )
        self.loss = nn.MSELoss()

    def forward(self, x, target):
        pred = self.module(x)
        return self.loss(pred, target)

def benchmark_module(module, input_shape=(256, 1, 29, 29), warmup=0, iters=1000):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    module = module.to(device)
    module.eval()

    x = torch.randn(input_shape).to(device)
    target = torch.randn((input_shape[0], 10)).to(device)

    # Warmup
    for _ in range(warmup):
        _ = module(x, target)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ----------------------------------------
    # Real benchmark (with NVTX ranges)
    # ----------------------------------------

    if device.type == "cuda":
        torch.cuda.nvtx.range_push("benchmark_loop")

    start = time.time()
    for i in range(iters):
        if device.type == "cuda":
            torch.cuda.nvtx.range_push(f"iter_{i}")

        _ = module(x, target)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()  # iter end

    end = time.time()

    if device.type == "cuda":
        torch.cuda.nvtx.range_pop()  # benchmark_loop end

    total_time_sec = end - start
    avg_time_ms = (end - start) * 1000 / iters

    print("=" * 50)
    print(f"Device          : {device}")
    print(f"Input shape     : {input_shape}")
    print(f"Warmup steps    : {warmup}")
    print(f"Benchmark iters : {iters}")
    print(f"Total time      : {total_time_sec:.6f} sec")
    print(f"Avg time        : {avg_time_ms:.4f} ms / forward")
    print("=" * 50)

    return avg_time_ms


if __name__ == "__main__":
    net = MyModule()
    benchmark_module(net)
