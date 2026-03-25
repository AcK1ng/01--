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


// 用于保存各上下文长度的统计结果
struct BenchmarkMetrics {
    double mean_ms;
    double p50_ms;
    double p90_ms;
    double p99_ms;
};

// 工具函数：计算单次 1000 请求的延迟指标
BenchmarkMetrics calculate_metrics(const std::vector<double>& latencies) {
    std::vector<double> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());

    double mean = 0.0;
    for (double lat : sorted) mean += lat;
    mean /= sorted.size();

    auto get_percentile = [&](double p) {
        int idx = std::max(0, (int)(sorted.size() * p) - 1);
        return sorted[idx];
    };

    return {mean, get_percentile(0.50), get_percentile(0.90), get_percentile(0.99)};
}

// 保存最终汇总的 JSON
void save_context_results_to_json(const std::string& filepath, 
                                  const std::map<int, BenchmarkMetrics>& all_results) {
    std::ofstream out(filepath);
    if (!out.is_open()) return;

    out << "{\n";
    out << "  \"system\": \"ORS_Framework\",\n";
    out << "  \"data\": {\n";
    
    size_t count = 0;
    for (const auto& kv : all_results) {
        int seq_len = kv.first;
        BenchmarkMetrics m = kv.second;
        
        out << "    \"" << seq_len << "\": {\n";
        out << "      \"mean_ms\": " << m.mean_ms << ",\n";
        out << "      \"p50_ms\": " << m.p50_ms << ",\n";
        out << "      \"p90_ms\": " << m.p90_ms << ",\n";
        out << "      \"p99_ms\": " << m.p99_ms << "\n";
        out << "    }";
        if (++count < all_results.size()) out << ",";
        out << "\n";
    }
    
    out << "  }\n";
    out << "}\n";
    std::cout << "Context lengths benchmark saved to " << filepath << "\n";
}

void force_global_synchronize_debug() {
    for (int dev_idx = 0; dev_idx < 1; ++dev_idx) {
        c10::cuda::CUDAGuard g(dev_idx);
        cudaDeviceSynchronize();
    }
}

void save_results_to_json(const std::string& filepath, 
                          const std::string& system_name, 
                          const std::vector<double>& latencies_ms) {
    if (latencies_ms.empty()) return;

    // 复制一份并排序以计算百分位
    std::vector<double> sorted = latencies_ms;
    std::sort(sorted.begin(), sorted.end());

    double mean = 0.0;
    for (double lat : sorted) {
        mean += lat;
    }
    mean /= sorted.size();

    auto get_percentile = [&](double p) {
        int idx = std::max(0, (int)(sorted.size() * p) - 1);
        return sorted[idx];
    };

    double p50 = get_percentile(0.50);
    double p90 = get_percentile(0.90);
    double p99 = get_percentile(0.99);

    std::cout << "----------------------------------------\n";
    std::cout << "[" << system_name << "] End-to-End Latency:\n";
    std::cout << "Mean : " << std::fixed << std::setprecision(3) << mean << " ms\n";
    std::cout << "P50  : " << p50 << " ms\n";
    std::cout << "P90  : " << p90 << " ms\n";
    std::cout << "P99  : " << p99 << " ms\n";
    std::cout << "----------------------------------------\n";

    // 写入简单的 JSON 文件
    std::ofstream out(filepath);
    if (!out.is_open()) {
        LOG(ERROR) << "Failed to open " << filepath << " for writing.";
        return;
    }

    out << "{\n";
    out << "  \"system\": \"" << system_name << "\",\n";
    out << "  \"metrics\": {\n";
    out << "    \"mean_ms\": " << mean << ",\n";
    out << "    \"p50_ms\": " << p50 << ",\n";
    out << "    \"p90_ms\": " << p90 << ",\n";
    out << "    \"p99_ms\": " << p99 << "\n";
    out << "  },\n";
    out << "  \"raw_latencies_ms\":[";
    for (size_t i = 0; i < latencies_ms.size(); ++i) {
        out << latencies_ms[i] << (i == latencies_ms.size() - 1 ? "" : ", ");
    }
    out << "]\n";
    out << "}\n";
    
    std::cout << "Results saved to " << filepath << "\n";
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
    std::vector<int> context_lengths = {1, 4, 16, 32, 64, 128, 512, 1024, 2048, 4096, 8192, 16384};
    // std::vector<int> context_lengths = {1, 4, 16, 32, 64, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    // std::vector<int> context_lengths = {65536, 131072};
    //std::vector<int> context_lengths = {1024, 2048, 4096, 16384, 65536, 131072};
    int test_iters = 1000;
    std::map<int, BenchmarkMetrics> all_results;

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

        // 1. 预热
        warmup(master, model_id, move_inputs, 500);

        // 2. 正式测速 1000 次
        std::vector<double> latencies_ms;
        latencies_ms.reserve(test_iters);
        butil::Status ret;

        for (int i = 0; i < test_iters; i++) {
            scoped_refptr<ors::Graph> graph;
            ret = master->get_execute_graph(model_id, graph);
            if (!ret.ok()) break;
            
            master->fill_graph_with_inputs(graph, move_inputs);

            ors::SynchronizedClosure done;
            ors::g_total_infer_latency.Reset();
            ors::g_total_infer_latency.Start();
            
            master->run_graph(graph, &done);
            done.wait();
            force_global_synchronize_debug();
            
            ors::g_total_infer_latency.Stop();
            latencies_ms.push_back(ors::g_total_infer_latency.GetAccumulatedNs() / 1000000.0);

            graph->_registers.clear();
            graph->reset();
        }

        BenchmarkMetrics metrics = calculate_metrics(latencies_ms);
        all_results[seq_len] = metrics;
        std::cout << "Decode SeqLen " << seq_len << " | Mean: " << metrics.mean_ms << "ms | P99: " << metrics.p99_ms << "ms\n";
    }

    mkdir("../results", 0777); 
    save_context_results_to_json("../results/ors_decode_latency.json", all_results);

    master->stop();
    return 0;
}