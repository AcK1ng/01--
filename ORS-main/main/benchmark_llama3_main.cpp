#include <torch/script.h>
#include <torch/cuda.h>
#include <unistd.h>
#include <sys/stat.h>
#include "master/master.h"
#include "common/config.h"
#include "worker/engine.h"
#include <gflags/gflags.h>
#include "common/closure_helper.h"
#include "butil/logging.h"
#include "common/utils.h"
#include "common/timing.h"
#include "common/cuda_profile.h"
#include "master/graph.h"
#include "master/latency_estimate.h"
#include <nvToolsExt.h>

namespace ors {
extern LatencyEstimator g_latency_est;
}

DECLARE_string(flagfile);

void force_global_synchronize_debug() {
    for (int dev_idx = 0; dev_idx < 1; ++dev_idx) {
        c10::cuda::CUDAGuard g(dev_idx);
        cudaDeviceSynchronize();
    }
}

c10::IValue moveIValueToDevice(const c10::IValue& v, c10::Device device) {
    CHECK(v.isTensor() || v.isTuple());
    if (v.isTensor()) {
        return v.toTensor().to(device);
    } else {
        std::vector<c10::IValue> new_elems;
        for (auto& e : v.toTuple()->elements()) {
            new_elems.push_back(moveIValueToDevice(e, device));
        }
        return c10::IValue(c10::ivalue::Tuple::create(new_elems));
    }
}

void warmup(scoped_refptr<ors::Master> master, std::string model_id, 
            std::vector<torch::jit::IValue>& inputs, int iters, c10::Device device) {
    std::deque<scoped_refptr<ors::Graph>> graphs;
    butil::Status ret;
    for (int i = 0; i < iters; ++i) {
        scoped_refptr<ors::Graph> graph;
        ret = master->prepare_graph(model_id, graph);
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return;
        }
        graphs.push_back(graph);
    }

    std::vector<torch::jit::IValue> move_inputs;
    for (auto input : inputs) {
        move_inputs.push_back(moveIValueToDevice(input, device));
    }
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePush("warm_up");
    #endif
    for (int i = 0; i < iters; i++) {
        #ifdef ENABLE_NVTX_DEBUG
        // 为每一轮创建唯一的标签
        std::string iter_label = "iter_" + std::to_string(i);
        nvtxRangePush(iter_label.c_str());
        #endif
        ors::SynchronizedClosure done;
        ret = master->submit_infer_request_with_graph(graphs[i], move_inputs, &done);
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return;
        }
        done.wait();
        #ifdef ENABLE_NVTX_DEBUG
        nvtxRangePop();  // 结束当前迭代的范围
        #endif
    }
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePop();
    #endif
}

std::vector<torch::jit::IValue> move_input_to_device(std::vector<torch::jit::IValue>& inputs, c10::Device device) {
    std::vector<torch::jit::IValue> move_inputs;
    for (auto input : inputs) {
        move_inputs.push_back(moveIValueToDevice(input, device));
    }
    return move_inputs;
}

void run_seq_request(scoped_refptr<ors::Master> master, std::string model_id, 
            std::vector<torch::jit::IValue>& inputs, int iters) {
    std::deque<scoped_refptr<ors::Graph>> graphs;
    butil::Status ret;
    for (int i = 0; i < iters; ++i) {
        scoped_refptr<ors::Graph> graph;
        ret = master->prepare_graph(model_id, graph);
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return;
        }
        master->fill_graph_with_inputs(graph, inputs);
        graphs.push_back(graph);
    }
    std::cout << "begin to execute graph...." << std::endl;
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePush("run_seq");
    #endif
    uint64_t start_ns = butil::cpuwide_time_ns();
    for (int i = 0; i < iters; i++) {
        // #ifdef ENABLE_NVTX_DEBUG
        // 为每一轮创建唯一的标签
        std::string iter_label = "iter_" + std::to_string(i);
        nvtxRangePush(iter_label.c_str());
        // #endif
        ors::SynchronizedClosure done;
        ret = master->run_graph(graphs[i], &done);
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return;
        }
        done.wait();
        force_global_synchronize_debug();
        // #ifdef ENABLE_NVTX_DEBUG
        nvtxRangePop();  // 结束当前迭代的范围
        // #endif
    }
    uint64_t latency_ns = butil::cpuwide_time_ns() - start_ns;
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePop();
    #endif
    std::cout << "total infer time(ns): " << latency_ns << "\n";
}

void run_concurrent_request(scoped_refptr<ors::Master> master, std::string model_id, 
            std::vector<torch::jit::IValue>& inputs, int iters) {
    std::deque<scoped_refptr<ors::Graph>> graphs;
    butil::Status ret;
    for (int i = 0; i < iters; ++i) {
        scoped_refptr<ors::Graph> graph;
        ret = master->prepare_graph(model_id, graph);
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return;
        }
        graphs.push_back(graph);
    }
    // sleep(10);
    ors::SynchronizedClosure sync_done;
    auto closures = new ors::ClosureGroup<>(iters, &sync_done);
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePush("run_concurrent");
    #endif
    force_global_synchronize_debug();
    for (int i = 0; i < iters; i++) {
        #ifdef ENABLE_NVTX_DEBUG
        // 为每一轮创建唯一的标签
        std::string iter_label = "iter_" + std::to_string(i);
        nvtxRangePush(iter_label.c_str());
        #endif
        ret = master->submit_infer_request_with_graph(graphs[i], inputs, closures->sub_done(i));
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return;
        }
        #ifdef ENABLE_NVTX_DEBUG
        nvtxRangePop();  // 结束当前迭代的范围
        #endif
    }
    uint64_t start_ms = butil::cpuwide_time_ms();
    // uint64_t start_ns = butil::cpuwide_time_ns();
    sync_done.wait();
    CHECK(sync_done.status().ok());
    // force_global_synchronize_debug();
    // uint64_t latency_ns = butil::cpuwide_time_ns() - start_ns;
    uint64_t latency_ms = butil::cpuwide_time_ms() - start_ms;
    for (int i = 0; i < iters; i++) {
        graphs[i]->reset();
    }
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePop();
    #endif
    std::cout << "total infer time(ms): " << latency_ms << "\n";
}

auto device0 = torch::Device(torch::kCUDA, 0);

// 模拟 Params 类
struct ModelParams {
    int max_seq_len = 16*1024;
    int n_kv_heads = 8;
    int dim = 4096;
    int n_heads = 32;
    int n_layers = 32;
};

class KVGenerator {
public:
    explicit KVGenerator(ModelParams params) : params_(params) {}

    // 内部函数：生成 KV Caches 列表
    std::vector<torch::Tensor> _generate_kvcaches(
        int max_seq_len, 
        int n_kv_heads, 
        int head_dim, 
        int n_layers
    ) {
        std::vector<torch::Tensor> caches;
        caches.reserve(n_layers); // 预分配空间，优化性能

        for (int i = 0; i < n_layers; ++i) {
            // Python: torch.empty(2, max_seq_len, n_kv_heads, head_dim)
            // C++: torch::empty({2, max_seq_len, ...})
            // 注意：这里默认是 float32，如果需要 half/fp16，加上 options
            auto cache = torch::empty(
                {2, max_seq_len, n_kv_heads, head_dim}, 
                torch::TensorOptions() // 可以在这里指定 .dtype(torch::kHalf).device(torch::kCUDA)
            );
            caches.push_back(cache);
        }
        return caches;
    }

    // 公开接口：处理默认参数逻辑
    // 使用 std::optional<int> 模拟 Python 的 Optional[int] = None
    std::vector<torch::Tensor> generate_kvcaches(std::optional<int> max_seq_len = std::nullopt) {
        int seq_len = max_seq_len.value_or(params_.max_seq_len);
        
        int head_dim = params_.dim / params_.n_heads; // 整数除法

        return _generate_kvcaches(
            seq_len, 
            params_.n_kv_heads, 
            head_dim, 
            params_.n_layers
        );
    }

private:
    ModelParams params_;
};

void display_input_devices(std::vector<torch::jit::IValue>& inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].isTensor()) {
            std::cout << "Input " << i << " is on device: " << inputs[i].toTensor().device() << std::endl;
        } else if (inputs[i].isTuple()) {
            std::cout << "Input " << i << " is a tuple." << std::endl;
            for (const auto& element : inputs[i].toTuple()->elements()) {
                if (element.isTensor()) {
                    std::cout << "  Element is on device: " << element.toTensor().device() << std::endl;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    struct stat st;
    if (0 == stat("conf/master.conf", &st) && FLAGS_flagfile.empty()) {
        FLAGS_flagfile = "conf/master.conf";
    }
    // parse command line
    google::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_minloglevel = google::INFO;
    if (0 != ors::init_log()) {
        fprintf(stderr, "init log failed\n");
        return -1;
    }

    // start engine
    ors::Engine* engine = new ors::Engine();
    engine->start();
    // ors::ProfileConfig cfg;
    // cfg.profile_data_path = "../data/data.profile";
    // ors::g_latency_est.Init(cfg);
    // ors::IScheduler* scheduler = new ors::HEFTScheduler(engine, &ors::g_latency_est);
    ors::IScheduler* scheduler = new ors::RoundRobinScheduler(engine);
    // scheduler->start();
    scoped_refptr<ors::Master> master = new ors::Master(engine, scheduler);
    std::string model_id = "llama3";
    std::map<std::string, std::string> sub_modules;
    sub_modules["submod_op"] = "/home/wangyulong/ORS/GCG/UserProgram/llama3_infer_TS/ts_fw_submods_submod_op.txt";
    master->register_model(model_id, "/home/wangyulong/ORS/GCG/UserProgram/llama3_infer_TS/ts_fw_root.txt", sub_modules);

    std::vector<torch::jit::IValue> inputs;
    // get fw params
    torch::jit::Module module = torch::jit::load("/home/wangyulong/ORS/GCG/UserProgram/llama3_infer_TS/fw_params_container.pt");
    c10::IValue ivalue = module.attr("buffers");
    CHECK(ivalue.isTuple());
    for (auto element : ivalue.toTuple()->elements()) {
        for (auto& e : element.toTuple()->elements()) {
            std::cout << "fw param element device: " << e.toTensor().device() << std::endl;
            // new_elems.push_back(moveIValueToDevice(e, device));
        }
        // std::cout << "fw param element device: " << element.toIValue().toTensor().device() << std::endl;
        inputs.push_back(element.toIValue());
    }
    // get fw buffers
    module = torch::jit::load("/home/wangyulong/ORS/GCG/UserProgram/llama3_infer_TS/fw_buffer_container.pt");
    ivalue = module.attr("buffers");
    CHECK(ivalue.isTuple());
    for (auto element : ivalue.toTuple()->elements()) {
        std::cout << "fw buffer element device: " << element.toTensor().device() << std::endl;
        // for (auto& e : element.toTuple()->elements()) {
        //     std::cout << "fw param element device: " << e.toTensor().device() << std::endl;
        //     // new_elems.push_back(moveIValueToDevice(e, device));
        // }
        // std::cout << "fw buffer element device: " << element.toIValue().toTensor().device() << std::endl;
        inputs.push_back(element.toIValue());
    }

    ModelParams params;
    KVGenerator generator(params);
    auto kvcaches = generator.generate_kvcaches(128);
    for (auto& cache : kvcaches) {
        std::cout << "kvcache device: " << cache.device() << std::endl;
        // inputs.push_back(cache);
    }
    inputs.insert(inputs.end(), kvcaches.begin(), kvcaches.end());
    auto tokens = torch::zeros({1, 1},torch::TensorOptions().dtype(torch::kLong));
    inputs.push_back(tokens);
    // std::cout << "tokens device: " << tokens.device() << std::endl;
    auto dummy_tensor__for__start_pos = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat));
    inputs.push_back(dummy_tensor__for__start_pos);
    
    
    // get example inputs
    // module = torch::jit::load("/home/wangyulong/ORS/GCG/UserProgram/llama3_infer_TS/example_inputs_container.pt");
    // ivalue = module.attr("buffers");
    // CHECK(ivalue.isTuple());
    // int i = 1;
    // for (auto element : ivalue.toTuple()->elements()) {
    //     std::cout << "fw input element device: " << element.toTensor().device() << std::endl;
    //     inputs.push_back(element.toIValue());
    //     if (i > 32) {
    //         std::cout << i  << " " << element.toTensor() << std::endl;
    //     }
    //     i++;
    // }
    



    c10::Device device(c10::kCUDA, 0);
    auto move_inputs = move_input_to_device(inputs, device);
    // 确保数据全部迁移到设备上了
    force_global_synchronize_debug();
    // display_input_devices(move_inputs);
    // run_seq_request(master, model_id, move_inputs, 1);
    // run_seq_request(master, model_id, move_inputs, 50);
    // run_concurrent_request(master, model_id, move_inputs,100);
    // run_concurrent_request(master, model_id, move_inputs,10);
    // run_concurrent_request(master, model_id, move_inputs,10);
    run_seq_request(master, model_id, move_inputs, 5);
    // force_global_synchronize_debug();
    {
        
        // run_concurrent_request(master, model_id, move_inputs, 1000);
        run_seq_request(master, model_id, move_inputs, 1000);
        force_global_synchronize_debug();
    }



    master->stop();
    return 0;
}