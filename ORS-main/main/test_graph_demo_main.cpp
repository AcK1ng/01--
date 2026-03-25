#include "master/master.h"
#include "common/config.h"
#include "worker/engine.h"
#include <gflags/gflags.h>
#include "common/closure_helper.h"
#include "butil/logging.h"
#include <nvToolsExt.h>
// #include "cuda/printer.cu"

void sync_device(int device_index) {
    c10::cuda::CUDAGuard guard(device_index);
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    // parse command line
    google::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_minloglevel = google::ERROR;
    if (0 != ors::init_log()) {
        fprintf(stderr, "init log failed\n");
        return -1;
    }

    // 定义设备
    torch::Device dev0(torch::kCUDA, 0);
    torch::Device dev1(torch::kCUDA, 1);

    // input
    at::Tensor x = torch::ones({2, 3}, dev0);
    at::Tensor y = torch::ones({2, 3}, dev0);
    at::Tensor z = torch::ones({2,3}, dev0);

    at::Tensor expected_output = (x + y + z).cpu(); 
    LOG(INFO) << "Expected Output (CPU): " << expected_output;

    // 预热
    std::cout << "正在预热 CUDA..." << std::endl;
    for (int i = 0; i < 3; ++i) { 
        c10::cuda::CUDAGuard guard(0); // 切换上下文
        auto tmp = x.to(dev0) + y.to(dev0) + z.to(dev0);
    }
    for (int i = 0; i < 3; ++i) { 
        c10::cuda::CUDAGuard guard(1); // 切换上下文
        auto tmp = x.to(dev1) + y.to(dev1) + z.to(dev1);
        
    }
    sync_device(dev0.index());
    sync_device(dev1.index());

    // start engine
    ors::Engine* engine = new ors::Engine();
    engine->start();
    ors::IScheduler* scheduler = new ors::RoundRobinScheduler(engine);
    scheduler->start();
    scoped_refptr<ors::Master> master = new ors::Master(engine, scheduler);
    std::string model_id = "TorchScriptDemo";    
    master->register_model(model_id, "/home/wangyulong/ORS/test/my_test1/test_graph.txt");

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::jit::IValue(x));
    inputs.push_back(torch::jit::IValue(y));
    inputs.push_back(torch::jit::IValue(z));

    nvtxRangePush(">>> PHASE 1: PIPELINE (LibTorch API) <<<");
    std::vector<std::optional<torch::jit::IValue>> outputs;
    uint64_t req_id;
    std::vector<uint64_t> req_ids;
    int infer_count = 50;
    req_ids.resize(infer_count);
    for (int i = 0; i < infer_count; i++) {
        ors::SynchronizedClosure done;
        auto st = master->submit_infer_request(model_id, inputs, &done, req_ids[i]);
        if (!st.ok()) {
            LOG(ERROR) << "submit infer request failed!";
            return 0;
        }
        done.wait();
    }
    sync_device(dev0.index());
    sync_device(dev1.index());
    nvtxRangePop();
    nvtxRangePush(">>> PHASE 2: Compare results <<<");
    for (int i = 0; i < infer_count; i++) {
        outputs.clear();
        master->get_infer_result(model_id, req_ids[i], outputs);
        // master->get_infer_result(model_id, req_id, outputs);
        // --- 验证逻辑 ---
        if (outputs.empty() || !outputs[0].has_value()) {
            LOG(ERROR) << "Iter " << i << ": Output is empty or invalid!";
            continue;
        }
        // 取出 Tensor 并转到 CPU 进行比较
        at::Tensor actual_output = outputs[0].value().toTensor().cpu();

        // 使用 allclose 进行浮点数比较
        if (torch::allclose(actual_output, expected_output, 1e-3, 1e-3)) {
            // if (i == 0) {
                LOG(ERROR) << "Iter 0 Verified ✅: Result matches.";
            // }
            // success_count++;
        } else {
            LOG(ERROR) << "Iter " << i << " Verification FAILED ❌, req_id: " << req_ids[i];
            LOG(ERROR) << "Expected: " << expected_output;
            LOG(ERROR) << "Actual:   " << actual_output;
            // 打印差值
            LOG(ERROR) << "Max Diff: " << (actual_output - expected_output).abs().max().item<float>();
            break; // 遇到错误立刻停止，方便调试
        }
    }
    nvtxRangePop();



    master->stop();
    return 0;
}