#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAEvent.h> // 引入 PyTorch 的 Event 封装
#include <nvToolsExt.h>
#include <iostream>

// 辅助函数：同步指定的设备
void sync_device(int device_index) {
    c10::cuda::CUDAGuard guard(device_index);
    cudaDeviceSynchronize();
}

void inspect_dirty_tensor() {
    if (torch::cuda::device_count() < 2) return;
    
    torch::Device src_dev(torch::kCUDA, 0);
    torch::Device dst_dev(torch::kCUDA, 1);
    
    // 获取流
    auto stream_compute = at::cuda::getStreamFromPool(false, src_dev.index());
    auto stream_peek    = at::cuda::getStreamFromPool(false, src_dev.index()); // 用于偷看的流

    // 1. 初始化数据为 0
    torch::Tensor data;
    {
        at::cuda::CUDAStreamGuard guard(stream_compute);
        data = torch::zeros({4096, 4096}, torch::TensorOptions().device(src_dev));
    }
    
    // 2. 在 Compute 流上发射一个耗时任务
    // 任务：先做矩阵乘法（耗时），然后把 data 全变成 1
    {
        at::cuda::CUDAStreamGuard guard(stream_compute);
        
        // 耗时操作：让 GPU 忙一会儿
        auto temp = torch::matmul(data, data); 
        
        // 真正写入 1.0 的操作
        data.fill_(1.0); 
    }
    
    // 3. 在 Peek 流上尝试“偷看” data 的值
    // 此时 Compute 流上的 matmul 可能还在跑，fill_(1.0) 还没执行
    torch::Tensor peek_result;
    {
        at::cuda::CUDAStreamGuard guard(stream_peek);
        
        // 【关键】：这里故意不写 event.block(stream_peek)
        // 我们想看看没等待的时候，data 是多少
        
        // 把 data 搬运到 CPU 或者另一个 GPU
        // 使用 non_blocking=true 确保这行指令立刻发出，不等待 Compute 流
        peek_result = data.to(torch::kCPU, /*non_blocking=*/true);
    }
    
    // 4. 等待偷看结束，但不等待计算结束
    // 我们只同步 Peek 流，确保 peek_result 已经拿到了（不管拿的是旧值还是新值）
    {
        c10::cuda::CUDAGuard guard(src_dev.index());
        cudaStreamSynchronize(stream_peek.stream());
    }

    // 5. 打印结果
    // 理论上：
    // 如果 data.fill_(1.0) 还没执行，这里应该看到 0。
    // 如果执行完了，应该看到 1。
    float val = peek_result[0][0].item<float>();
    std::cout << "在 Kernel 未完成时偷看的值: " << val << std::endl;
    
    if (val == 0.0) {
        std::cout << "=> 成功捕获到了旧值！(Data Race 验证成功)" << std::endl;
    } else {
        std::cout << "=> 依然读到了新值 (GPU 计算太快了，或者提交指令太慢)" << std::endl;
    }

    // 最后清理现场
    cudaDeviceSynchronize();
}

void verify_scheme_logic() {
    // 0. 环境检查
    if (!torch::cuda::is_available() || torch::cuda::device_count() < 2) {
        std::cerr << "Error: 需要至少两个 GPU 才能运行此验证程序。" << std::endl;
        return;
    }

    // 定义设备
    torch::Device src_dev(torch::kCUDA, 0);
    torch::Device dst_dev(torch::kCUDA, 1);

    std::cout << "Source: " << src_dev << ", Destination: " << dst_dev << std::endl;

    // 1. 创建流 (Streams)
    // 必须确保拿到不同的流。getStreamFromPool 第二个参数是 priority，如果不传可能拿到相同的流。
    // 为了稳妥，我们直接从 StreamPool 获取，或者使用 getStreamFromPool 的机制
    auto src_compute_stream = at::cuda::getStreamFromPool(false, src_dev.index());
    auto src_tx_stream      = at::cuda::getStreamFromPool(false, src_dev.index());
    
    auto dst_rx_stream      = at::cuda::getStreamFromPool(false, dst_dev.index());
    auto dst_compute_stream = at::cuda::getStreamFromPool(false, dst_dev.index());

    // 2. 创建 Events (使用 at::cuda::CUDAEvent)
    // flag: cudaEventDisableTiming (对应 Python 的 blocking=False/disable_timing=True)
    at::cuda::CUDAEvent event_comp_done(cudaEventDisableTiming);
    at::cuda::CUDAEvent event_tx_done(cudaEventDisableTiming);
    at::cuda::CUDAEvent event_rx_ready(cudaEventDisableTiming);

    // 准备数据
    int64_t N = 4096;
    torch::Tensor data_src;
    
    // 初始化数据
    {
        at::cuda::CUDAStreamGuard guard(src_compute_stream);
        data_src = torch::randn({N, N}, torch::TensorOptions().device(src_dev));
    }

    // 预热
    std::cout << "正在预热 CUDA..." << std::endl;
    for (int i = 0; i < 3; ++i) { torch::matmul(data_src, data_src); }
    sync_device(src_dev.index());
    sync_device(dst_dev.index());

    std::cout << "开始执行并记录 Trace..." << std::endl;

    nvtxRangePush(">>> PHASE 1: PIPELINE (LibTorch API) <<<");

    torch::Tensor result_src;
    torch::Tensor data_dst_buffer;
    torch::Tensor final_result;

    // === 步骤 1: 源设备计算 ===
    {
        at::cuda::CUDAStreamGuard guard(src_compute_stream);
        nvtxRangePush("1. Src Compute");
        
        result_src = torch::matmul(data_src, data_src);
        
        // [Record]
        // 记录 event 到当前流 (src_compute_stream)
        event_comp_done.record(src_compute_stream);
        
        nvtxRangePop();

        // Overlap 验证
        nvtxRangePush("1.5 Overlap Check");
        torch::matmul(data_src, data_src); 
        nvtxRangePop();
    }

    // === 步骤 2: 源设备发送 (Tx) ===
    {
        at::cuda::CUDAStreamGuard guard(src_tx_stream);
        nvtxRangePush("2. Tx Wait & Copy");

        // [Wait]
        // 让当前流 (src_tx_stream) 等待 event_comp_done
        // PyTorch API 中使用 block() 方法来实现 wait_event
        event_comp_done.block(src_tx_stream);

        // P2P Copy
        data_dst_buffer = result_src.to(dst_dev, /*non_blocking=*/true);

        // [Record]
        event_tx_done.record(src_tx_stream);
        nvtxRangePop();
    }

    // === 步骤 3: 目标设备接收 (Rx) ===
    {
        at::cuda::CUDAStreamGuard guard(dst_rx_stream);
        nvtxRangePush("3. Rx Signal Wait");

        // [Wait]
        event_tx_done.block(dst_rx_stream);

        // [Record]
        event_rx_ready.record(dst_rx_stream);
        nvtxRangePop();
    }

    // === 步骤 4: 目标设备计算 ===
    {
        at::cuda::CUDAStreamGuard guard(dst_compute_stream);
        nvtxRangePush("4. Dst Compute");

        // [Wait]
        event_rx_ready.block(dst_compute_stream);

        // Compute
        final_result = torch::matmul(data_dst_buffer, data_dst_buffer);
        nvtxRangePop();
    }

    sync_device(src_dev.index());
    sync_device(dst_dev.index());
    nvtxRangePop();

    std::cout << "执行完成。正在验证..." << std::endl;

    // === 验证部分 ===
    nvtxRangePush(">>> PHASE 2: VERIFICATION <<<");
    torch::Tensor expected = torch::matmul(
        torch::matmul(data_src, data_src).to(dst_dev),
        torch::matmul(data_src, data_src).to(dst_dev)
    );
    sync_device(src_dev.index());
    sync_device(dst_dev.index());
    nvtxRangePop();

    if (torch::allclose(final_result, expected, 1e-3, 1e-2)) {
        std::cout << "✅ 验证成功！" << std::endl;
    } else {
        std::cout << "❌ 验证失败！" << std::endl;
        std::cout << "Max diff: " << (final_result - expected).abs().max().item<float>() << std::endl;
    }
}

void verify_my_logic() {
    // 0. 环境检查
    if (!torch::cuda::is_available() || torch::cuda::device_count() < 2) {
        std::cerr << "Error: 需要至少两个 GPU 才能运行此验证程序。" << std::endl;
        return;
    }

    // 定义设备
    torch::Device src_dev(torch::kCUDA, 0);
    torch::Device dst_dev(torch::kCUDA, 1);

    std::cout << "Source: " << src_dev << ", Destination: " << dst_dev << std::endl;

    // 1. 创建流 (Streams)
    // 必须确保拿到不同的流。getStreamFromPool 第二个参数是 priority，如果不传可能拿到相同的流。
    // 为了稳妥，我们直接从 StreamPool 获取，或者使用 getStreamFromPool 的机制
    auto src_compute_stream = at::cuda::getStreamFromPool(false, src_dev.index());
    auto src_tx_stream      = at::cuda::getStreamFromPool(false, src_dev.index());
    
    auto dst_tx_stream      = at::cuda::getStreamFromPool(false, dst_dev.index());
    auto dst_compute_stream = at::cuda::getStreamFromPool(false, dst_dev.index());

    // 2. 创建 Events (使用 at::cuda::CUDAEvent)
    // flag: cudaEventDisableTiming (对应 Python 的 blocking=False/disable_timing=True)
    at::cuda::CUDAEvent event_comp_done(cudaEventDisableTiming);
    at::cuda::CUDAEvent event_transfer_done(cudaEventDisableTiming);
    // at::cuda::CUDAEvent event_rx_ready(cudaEventDisableTiming);

    // 准备数据
    int64_t N = 4096;
    torch::Tensor data_src;
    
    // 初始化数据
    {
        at::cuda::CUDAStreamGuard guard(src_compute_stream);
        data_src = torch::randn({N, N}, torch::TensorOptions().device(src_dev));
    }

    // 预热
    std::cout << "正在预热 CUDA..." << std::endl;
    for (int i = 0; i < 3; ++i) { 
        at::cuda::CUDAStreamGuard guard(src_compute_stream);
        torch::matmul(data_src, data_src); 
    }
    for (int i = 0; i < 3; ++i) { 
        at::cuda::CUDAStreamGuard guard(dst_compute_stream);
        auto tmp_dst = data_src.to(dst_dev);
        torch::matmul(tmp_dst, tmp_dst); 
    }
    sync_device(src_dev.index());
    sync_device(dst_dev.index());

    std::cout << "开始执行并记录 Trace..." << std::endl;

    nvtxRangePush(">>> PHASE 1: PIPELINE (LibTorch API) <<<");

    torch::Tensor result_src;
    torch::Tensor data_dst_buffer;
    torch::Tensor final_result;

    torch::Tensor tmp;

    // === 步骤 1: 源设备计算 ===
    {
        at::cuda::CUDAStreamGuard guard(src_compute_stream);
        nvtxRangePush("1. Src Compute");
        
        result_src = torch::matmul(data_src, data_src);
        
        // [Record]
        // 记录 event 到当前流 (src_compute_stream)
        event_comp_done.record(src_compute_stream);
        
        nvtxRangePop();

        // Overlap 验证
        nvtxRangePush("1.5 Overlap Check");
        torch::matmul(data_src, data_src); 
        nvtxRangePop();
    }

    // === 步骤 2: 源设备发送 (Tx) ===
    {
        {
            at::cuda::CUDAStreamGuard guard(src_tx_stream);
            for (int i = 0; i < 100; ++i) {
                tmp = torch::matmul(data_src, data_src);
            }
        }
        at::cuda::setCurrentCUDAStream(src_tx_stream);
        at::cuda::setCurrentCUDAStream(dst_tx_stream);

        // [Wait]
        // 让当前流 (src_tx_stream) 等待 event_comp_done
        // PyTorch API 中使用 block() 方法来实现 wait_event
        event_comp_done.block(src_tx_stream);

        // P2P Copy
        data_dst_buffer = result_src.to(dst_dev, /*non_blocking=*/true);

        // [Record]
        event_transfer_done.record(dst_tx_stream);
        nvtxRangePop();
    }

    // === 步骤 3: 目标设备计算 ===
    {
        at::cuda::CUDAStreamGuard guard(dst_compute_stream);
        nvtxRangePush("3. Dst Compute");

        // [Wait]
        event_transfer_done.block(dst_compute_stream);

        // Compute
        final_result = torch::matmul(data_dst_buffer, data_dst_buffer);
        nvtxRangePop();
    }

    sync_device(src_dev.index());
    sync_device(dst_dev.index());
    nvtxRangePop();

    std::cout << "执行完成。正在验证..." << std::endl;

    // === 验证部分 ===
    nvtxRangePush(">>> PHASE 2: VERIFICATION <<<");
    torch::Tensor expected = torch::matmul(
        torch::matmul(data_src, data_src).to(dst_dev),
        torch::matmul(data_src, data_src).to(dst_dev)
    );
    sync_device(src_dev.index());
    sync_device(dst_dev.index());
    nvtxRangePop();

    if (torch::allclose(final_result, expected, 1e-3, 1e-2)) {
        std::cout << "✅ 验证成功！" << std::endl;
    } else {
        std::cout << "❌ 验证失败！" << std::endl;
        std::cout << "Max diff: " << (final_result - expected).abs().max().item<float>() << std::endl;
    }
}

int main() {
    verify_my_logic();
    return 0;
}