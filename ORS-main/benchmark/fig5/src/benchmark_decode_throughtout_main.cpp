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
#include "common/global_vars.h"

namespace ors {
extern LatencyEstimator g_latency_est;
extern Accumulator g_total_infer_latency;
}

DECLARE_string(flagfile);


struct ThroughputResult {
    double throughput_req_s;
};

std::map<std::string, std::map<int, ThroughputResult>> all_results;


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
            std::vector<torch::jit::IValue>& inputs, int iters) {
    std::deque<scoped_refptr<ors::Graph>> graphs;
    butil::Status ret;
    for (int i = 0; i < iters; ++i) {
        scoped_refptr<ors::Graph> graph;
        ret = master->get_execute_graph(model_id, graph);
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return;
        }
        master->fill_graph_with_inputs(graph, inputs);
        graphs.push_back(graph);
    }
    std::cout << "begin to execute graph...." << std::endl;
    uint64_t start_ns = butil::cpuwide_time_ns();
    for (int i = 0; i < iters; i++) {
        ors::SynchronizedClosure done;
        ret = master->run_graph(graphs[i], &done);
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return;
        }
        done.wait();
        force_global_synchronize_debug();
        // 图执行结束后，清理中间临时变量
        graphs[i]->_registers.clear();
        graphs[i]->reset();
    }
    uint64_t latency_ns = butil::cpuwide_time_ns() - start_ns;
    std::cout << "total infer time(ns): " << latency_ns << "\n";
}

std::vector<torch::jit::IValue> move_input_to_device(std::vector<torch::jit::IValue>& inputs, c10::Device device) {
    std::vector<torch::jit::IValue> move_inputs;
    for (auto input : inputs) {
        move_inputs.push_back(moveIValueToDevice(input, device));
    }
    return move_inputs;
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

void warm_up(scoped_refptr<ors::Master> master, std::string model_id, 
            std::vector<torch::jit::IValue>& inputs, int iters) {
    // 1. 准备图（保持不变）
    std::deque<scoped_refptr<ors::Graph>> graphs;
    butil::Status ret;
    for (int i = 0; i < iters; ++i) {
        scoped_refptr<ors::Graph> graph;
        ret = master->get_execute_graph(model_id, graph);
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return;
        }
        master->fill_graph_with_inputs(graph, inputs);
        graphs.push_back(graph);
    }

    // 定义批大小
    const int batch_size = 10;

    // 外层循环：按批次遍历
    for (int i = 0; i < iters; i += batch_size) {
        // 计算当前批次实际大小 (处理最后一批可能不满 10 个的情况)
        int current_batch_size = std::min(batch_size, iters - i);
        
        // 准备一组同步闭包 (SynchronizedClosure)
        // 注意：这里使用 vector 预分配，避免扩容导致的拷贝问题
        std::vector<ors::SynchronizedClosure> dones(current_batch_size);

        // --- Step 1: 批量提交 (Submit) ---
        for (int j = 0; j < current_batch_size; ++j) {
            int graph_idx = i + j;
            // 传入对应的 done 对象地址
            ret = master->run_graph(graphs[graph_idx], &dones[j]);
            if (!ret.ok()) {
                LOG(ERROR) << "run graph failed, index: " << graph_idx << ", ret: " << ret;
                return;
            }
        }
        
        for (int j = 0; j < current_batch_size; ++j) {
            dones[j].wait();
        }
        
        // 确保 GPU 上这批任务也都跑完了
        force_global_synchronize_debug();

        // --- Step 3: 批量清理 (Cleanup) ---
        for (int j = 0; j < current_batch_size; ++j) {
            int graph_idx = i + j;
            graphs[graph_idx]->_registers.clear();
            graphs[graph_idx]->reset();
        }
    }
}


void run_throughout_benchmark(scoped_refptr<ors::Master> master, std::string model_id, 
            std::vector<torch::jit::IValue>& inputs, int iters, int seq_len, int batch_size = 1) {
    // 1. 准备图（保持不变）
    std::deque<scoped_refptr<ors::Graph>> graphs;
    butil::Status ret;
    for (int i = 0; i < iters; ++i) {
        scoped_refptr<ors::Graph> graph;
        ret = master->get_execute_graph(model_id, graph);
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return;
        }
        master->fill_graph_with_inputs(graph, inputs);
        graphs.push_back(graph);
    }

    std::cout << "begin to execute graph (Batch Mode)...." << std::endl;
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePush("run_seq_batch");
    #endif
    
    uint64_t start_ns = butil::cpuwide_time_ns();

    // 定义批大小
    // const int batch_size = 10;
    std::string batch_key = "batch_" + std::to_string(batch_size);
    // 外层循环：按批次遍历
    for (int i = 0; i < iters; i += batch_size) {
        // 计算当前批次实际大小 (处理最后一批可能不满 10 个的情况)
        int current_batch_size = std::min(batch_size, iters - i);
        
        // 准备一组同步闭包 (SynchronizedClosure)
        // 注意：这里使用 vector 预分配，避免扩容导致的拷贝问题
        std::vector<ors::SynchronizedClosure> dones(current_batch_size);

        // --- Step 1: 批量提交 (Submit) ---
        for (int j = 0; j < current_batch_size; ++j) {
            int graph_idx = i + j;
            // 传入对应的 done 对象地址
            ret = master->run_graph(graphs[graph_idx], &dones[j]);
            if (!ret.ok()) {
                LOG(ERROR) << "run graph failed, index: " << graph_idx << ", ret: " << ret;
                return;
            }
        }

        // --- Step 2: 批量等待 (Wait) ---
        // 注意：这里的计时逻辑记录的是 "这批请求全部完成的总耗时"
        ors::g_total_infer_latency.Start();
        
        for (int j = 0; j < current_batch_size; ++j) {
            dones[j].wait();
        }
        
        // 确保 GPU 上这批任务也都跑完了
        force_global_synchronize_debug();
        
        ors::g_total_infer_latency.Stop();

        // --- Step 3: 批量清理 (Cleanup) ---
        for (int j = 0; j < current_batch_size; ++j) {
            int graph_idx = i + j;
            graphs[graph_idx]->_registers.clear();
            graphs[graph_idx]->reset();
        }
    }

    uint64_t latency_ns = butil::cpuwide_time_ns() - start_ns;
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePop();
    #endif
    
    // 5. 计算并打印吞吐量
    double total_latency_s = ors::g_total_infer_latency.GetAccumulatedNs() / 1e9;
    double throughput = (total_latency_s > 0) ? (iters / total_latency_s) : 0.0;
    double avg_latency_ms = (iters > 0) ? (ors::g_total_infer_latency.GetAccumulatedNs() / 1e6 / iters) : 0.0;
    all_results[batch_key][seq_len] = {throughput};
    std::cout << "----------------------------------------\n";
    std::cout << "Throughput Test Results:\n";
    std::cout << "  Total Requests: " << iters << "\n";
    std::cout << "  Total Time:     " << std::fixed << std::setprecision(4) << total_latency_s << " s\n";
    std::cout << "  Throughput:     " << std::fixed << std::setprecision(2) << throughput << " req/s\n";
    std::cout << "  Avg Latency:    " << std::fixed << std::setprecision(4) << avg_latency_ms << " ms/req\n";
    std::cout << "----------------------------------------\n";
}

void save_throughput_json(const std::string& filepath, 
                          const std::map<std::string, std::map<int, ThroughputResult>>& all_results) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << " for writing.\n";
        return;
    }

    // 设置浮点数输出格式，保留两位小数，让 JSON 看起来更整洁
    out << std::fixed << std::setprecision(2);

    out << "{\n";
    out << "  \"system\": \"ORS_Framework\",\n";
    out << "  \"benchmark\": \"Decode_Throughput\",\n";
    out << "  \"data\": {\n";
    
    size_t batch_count = 0;
    // 外层循环：遍历不同的 Batch Size (例如 "batch_1", "batch_10")
    for (const auto& batch_kv : all_results) {
        const std::string& batch_key = batch_kv.first;
        const auto& ctx_results = batch_kv.second;

        out << "    \"" << batch_key << "\": {\n";
        
        size_t ctx_count = 0;
        // 内层循环：遍历当前 Batch Size 下的不同 Context Length (例如 1, 128, 1024)
        for (const auto& ctx_kv : ctx_results) {
            int seq_len = ctx_kv.first;
            double throughput = ctx_kv.second.throughput_req_s;
            
            // 写入单个 context length 的数据
            out << "      \"" << seq_len << "\": { \"throughput_req_s\": " << throughput << " }";
            
            // 处理 JSON 逗号逻辑 (除了最后一个元素外，都需要逗号)
            if (++ctx_count < ctx_results.size()) {
                out << ",";
            }
            out << "\n";
        }
        
        out << "    }";
        // 处理外层 JSON 逗号逻辑
        if (++batch_count < all_results.size()) {
            out << ",";
        }
        out << "\n";
    }
    
    out << "  }\n";
    out << "}\n";

    std::cout << "Throughput benchmark results successfully saved to " << filepath << "\n";
}

int main(int argc, char** argv) {
    struct stat st;
    if (0 == stat("conf/master.conf", &st) && FLAGS_flagfile.empty()) {
        FLAGS_flagfile = "conf/master.conf";
    }
    // parse command line
    google::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_minloglevel = google::ERROR;
    if (0 != ors::init_log()) {
        fprintf(stderr, "init log failed\n");
        return -1;
    }

    // start engine
    ors::Engine* engine = new ors::Engine();
    engine->start();
    ors::IScheduler* scheduler = new ors::RoundRobinScheduler(engine);
    scoped_refptr<ors::Master> master = new ors::Master(engine, scheduler);
    std::string model_id = "llama3";
    std::map<std::string, std::string> sub_modules;
    sub_modules["submod_op"] = "/home/wangyulong/ORS/GCG/UserProgram/llama3_infer_TS/ts_fw_submods_submod_op.txt";
    master->register_model(model_id, "/home/wangyulong/ORS/GCG/UserProgram/llama3_infer_TS/ts_fw_root.txt", sub_modules);
    master->load_weights(model_id, "/home/wangyulong/ORS/GCG/UserProgram/llama3_infer_TS/fw_params_container.pt");
    master->load_buffers(model_id, "/home/wangyulong/ORS/GCG/UserProgram/llama3_infer_TS/fw_buffer_container.pt");

    c10::Device device(c10::kCUDA, 0);
    
    // 加入了长度 1
    // std::vector<int> context_lengths = {1};
    std::vector<int> context_lengths = {1, 4, 16, 32, 64, 128, 512, 1024, 2048, 4096};
    // std::vector<int> context_lengths = {1, 4, 16, 32, 64, 128, 512, 1024, 2048, 4096, 8192, 16384};
    // std::vector<int> context_lengths = {1, 4, 16, 32, 64, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    // std::vector<int> context_lengths = {65536, 131072};
    //std::vector<int> context_lengths = {1024, 2048, 4096, 16384, 65536, 131072};

    ModelParams params;
    KVGenerator generator(params);
    
    // // 统一分配最大容量 2048 的 KV Cache
    // int max_kv_capacity = 262144;
    // 统一分配128k个token+1个token的KVCache,以支持所有长度的测试
    int max_kv_capacity = 131072 + 1;
    auto kvcaches = generator.generate_kvcaches(max_kv_capacity);
    std::cout << "[ORS] Starting DECODE benchmark...\n";
    for (int seq_len : context_lengths) {
        std::cout << "\n--- Testing Decode at Context Length: " << seq_len << " ---\n";

        std::vector<torch::jit::IValue> inputs;
        // 把提前分配好的 KV Caches 放入输入中
        inputs.insert(inputs.end(), kvcaches.begin(), kvcaches.end());
        
        // 关键修改：动态改变 input tokens 的 sequence length 维度
        auto tokens = torch::zeros({1, 1}, torch::TensorOptions().dtype(torch::kLong));
        inputs.push_back(tokens);
        
        auto dummy_tensor = torch::empty({seq_len}, torch::TensorOptions().dtype(torch::kFloat));
        inputs.push_back(dummy_tensor);

        auto move_inputs = move_input_to_device(inputs, device);
        force_global_synchronize_debug();


        warm_up(master, model_id, move_inputs, 10);
        ors::g_total_infer_latency.Reset();
        force_global_synchronize_debug();
        for (int batch_size : {1, 10}) {
            std::cout << "\n=== Running Throughput Benchmark with Batch Size: " << batch_size << " ===\n";
            ors::g_total_infer_latency.Reset();
            run_throughout_benchmark(master, model_id, move_inputs, 1000, seq_len, batch_size);
            force_global_synchronize_debug();
        }
    }

    // mkdir("../results", 0777); 
    // save_context_results_to_json("../results/ors_decode_latency.json", all_results);

    save_throughput_json("../results/throughput_results.json", all_results);

    master->stop();
    return 0;
}