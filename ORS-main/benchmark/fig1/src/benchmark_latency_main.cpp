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

void run_macro_seq_benchmark(scoped_refptr<ors::Master> master, std::string model_id, 
                           std::vector<torch::jit::IValue>& inputs,
                           int num_samples, 
                           int reqs_per_sample, 
                           const std::string& output_json_path) {
    std::vector<double> latencies_ms;
    latencies_ms.reserve(num_samples);
    butil::Status ret;

    std::cout << "[ORS Benchmark] Running macro benchmark (" << num_samples 
              << " samples x " << reqs_per_sample << " requests)...\n";

    for (int i = 0; i < num_samples; i++) {
        // --- 记录 1000 个请求的总起始时间 ---
        
        // 串行执行 1000 个请求
        for (int j = 0; j < reqs_per_sample; j++) {
            ors::g_total_infer_latency.Reset();
            scoped_refptr<ors::Graph> graph;
            ret = master->get_execute_graph(model_id, graph);
            if (!ret.ok()) {
                LOG(ERROR) << "prepare graph failed, ret: " << ret;
                return;
            }
            master->fill_graph_with_inputs(graph, inputs);

            ors::SynchronizedClosure done;
            ors::g_total_infer_latency.Start();
            ret = master->run_graph(graph, &done);
            if (!ret.ok()) return;
            
            done.wait();
            
            // 每次请求结束都强制同步，保证严格的串行行为
            force_global_synchronize_debug();

            ors::g_total_infer_latency.Stop();
            double sample_latency_ms = ors::g_total_infer_latency.GetAccumulatedNs() / 1000000.0;
            latencies_ms.push_back(sample_latency_ms);
            // 及时清理临时变量，防止连续 1000 次请求把内存撑爆
            graph->_registers.clear();
            graph->reset();
        }
    }

    save_results_to_json(output_json_path, "ORS_Framework", latencies_ms);
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

void run_seq_request(scoped_refptr<ors::Master> master, std::string model_id, 
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
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePush("run_seq");
    #endif
    uint64_t start_ns = butil::cpuwide_time_ns();
    for (int i = 0; i < iters; i++) {
        // #ifdef ENABLE_NVTX_DEBUG
        // 为每一轮创建唯一的标签
        // std::string iter_label = "iter_" + std::to_string(i);
        // nvtxRangePush(iter_label.c_str());
        // #endif
        ors::SynchronizedClosure done;
        // nvtxRangePush("run_graph");
        ret = master->run_graph(graphs[i], &done);
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return;
        }
        // nvtxRangePop();
        ors::g_total_infer_latency.Start();
        done.wait();
        force_global_synchronize_debug();
        ors::g_total_infer_latency.Stop();
        // 图执行结束后，清理中间临时变量
        graphs[i]->_registers.clear();
        graphs[i]->reset();
        // #ifdef ENABLE_NVTX_DEBUG
        // nvtxRangePop();  // 结束当前迭代的范围
        // #endif
    }
    uint64_t latency_ns = butil::cpuwide_time_ns() - start_ns;
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePop();
    #endif
    std::cout << "total infer time(ns): " << latency_ns << "\n";
    std::cout << "Accumulated infer time(ns):" << ors::g_total_infer_latency.GetAccumulatedNs() << "\n";
}

void run_concurrent_request(scoped_refptr<ors::Master> master, std::string model_id, 
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

    std::cout << "begin to execute graph (Batch Mode)...." << std::endl;
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePush("run_seq_batch");
    #endif
    
    uint64_t start_ns = butil::cpuwide_time_ns();

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

    std::cout << "total infer time(ns): " << latency_ns << "\n";
    // 注意：这里的 Accumulated infer time 现在是 "所有批次等待时间的总和"
    std::cout << "Accumulated batch wait time(ns):" << ors::g_total_infer_latency.GetAccumulatedNs() << "\n";
    std::cout << "Average Latency per Request (Approx): " << (double)latency_ns / iters << " ns\n";
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
    master->load_weights(model_id, "/home/wangyulong/ORS/GCG/UserProgram/llama3_infer_TS/fw_params_container.pt");
    master->load_buffers(model_id, "/home/wangyulong/ORS/GCG/UserProgram/llama3_infer_TS/fw_buffer_container.pt");


    // sleep(60);

    ModelParams params;
    KVGenerator generator(params);
    auto kvcaches = generator.generate_kvcaches(128);
    for (auto& cache : kvcaches) {
        std::cout << "kvcache device: " << cache.device() << std::endl;
        // inputs.push_back(cache);
    }
    std::vector<torch::jit::IValue> inputs;
    inputs.insert(inputs.end(), kvcaches.begin(), kvcaches.end());
    auto tokens = torch::zeros({1, 1},torch::TensorOptions().dtype(torch::kLong));
    inputs.push_back(tokens);
    // std::cout << "tokens device: " << tokens.device() << std::endl;
    auto dummy_tensor__for__start_pos = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat));
    inputs.push_back(dummy_tensor__for__start_pos);

    c10::Device device(c10::kCUDA, 0);
    auto move_inputs = move_input_to_device(inputs, device);
    // 确保数据全部迁移到设备上了
    force_global_synchronize_debug();
    

    // 1. 强制预热
    std::cout << "[ORS] Warming up for 10 iterations...\n";
    warmup(master, model_id, move_inputs, 500);
    force_global_synchronize_debug();
    // 2. 确保存档目录存在
    mkdir("../results", 0777); 
    // 3. 正式测速，跑1000次，保存为 JSON
    // 执行实验：1000 次外层循环，每次测 1000 个串行请求的总时间
    run_macro_seq_benchmark(master, model_id, move_inputs, 1, 1000, "../results/ors_latency.json");

    master->stop();
    return 0;
}