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
#include "master/graph.h"
#include "master/latency_estimate.h"
#include <nvToolsExt.h>

namespace ors {
extern LatencyEstimator g_latency_est;
}

DECLARE_string(flagfile);

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

bool compare_ivalues_cpu(const torch::jit::IValue& v1, const torch::jit::IValue& v2, double atol=1e-3, double rtol=1e-3) {
    if (v1.tagKind() != v2.tagKind()) return false;

    if (v1.isTensor()) {
        // 核心：转到 CPU 进行比较
        at::Tensor t1 = v1.toTensor().cpu();
        at::Tensor t2 = v2.toTensor().cpu();
        
        // 形状检查
        if (t1.sizes() != t2.sizes()) return false;
        
        // 数值检查 (allclose)
        if (!torch::allclose(t1, t2, rtol, atol)) return false;
        
        return true;
    }

    if (v1.isTuple()) {
        auto elems1 = v1.toTuple()->elements();
        auto elems2 = v2.toTuple()->elements();
        if (elems1.size() != elems2.size()) return false;
        for (size_t i = 0; i < elems1.size(); ++i) {
            if (!compare_ivalues_cpu(elems1[i], elems2[i], atol, rtol)) return false;
        }
        return true;
    }
    
    // 基础类型直接比较
    if (v1.isInt()) return v1.toInt() == v2.toInt();
    if (v1.isDouble()) return std::abs(v1.toDouble() - v2.toDouble()) < atol;
    if (v1.isBool()) return v1.toBool() == v2.toBool();
    
    // 其他类型默认通过 (或按需扩展)
    return true;
}

void force_global_synchronize_debug() {
    // std::set<int> devices_to_sync;

    // // 1. 收集当前算子所在的设备
    // devices_to_sync.insert(op->device_idx());

    // // 2. 收集所有父算子所在的设备 (生产者)
    // for (auto& parent : op->get_parents()) {
    //     if (parent && parent->device_type() == c10::DeviceType::CUDA) {
    //         devices_to_sync.insert(parent->device_idx());
    //     }
    // }

    // 3. 遍历并强制同步每一个设备
    // 这会让 CPU 停下来，直到这些卡上的所有任务全部彻底跑完
    for (int dev_idx = 0; dev_idx < 2; ++dev_idx) {
        c10::cuda::CUDAGuard g(dev_idx);
        cudaDeviceSynchronize();
        // LOG(INFO) << "Force synchronized device: " << dev_idx;
    }
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
        #ifdef ENABLE_NVTX_DEBUG
        // 为每一轮创建唯一的标签
        std::string iter_label = "iter_" + std::to_string(i);
        nvtxRangePush(iter_label.c_str());
        #endif
        ors::SynchronizedClosure done;
        ret = master->run_graph(graphs[i], &done);
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return;
        }
        done.wait();
        #ifdef ENABLE_NVTX_DEBUG
        nvtxRangePop();  // 结束当前迭代的范围
        #endif
    }
    uint64_t latency_ns = butil::cpuwide_time_ns() - start_ns;
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePop();
    #endif
    std::cout << "total infer time(ns): " << latency_ns << "\n";
    // force_global_synchronize_debug();
    // std::vector<std::optional<torch::jit::IValue>> outputs;
    // master->get_graph_output(graphs[0], outputs);
    // for (int i = 1; i < iters; i++) {
    //     std::vector<std::optional<torch::jit::IValue>> curr_outputs;
    //     master->get_graph_output(graphs[i], curr_outputs);
    //     CHECK(outputs.size() == curr_outputs.size());
    //     for (size_t j = 0; j < outputs.size(); j++) {
    //         CHECK(outputs[j].has_value() && curr_outputs[j].has_value());
    //         bool equal = compare_ivalues_cpu(outputs[j].value(), curr_outputs[j].value());
    //         if (!equal) {
    //             LOG(ERROR) << "Output mismatch at iteration " << i << ", output index " << j;
    //         }
    //         CHECK(equal);
    //     }
    // }
    // torch::jit::IValue baseline_gpu = master->get_graph_output(graphs[0], 0);
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

    ors::SynchronizedClosure sync_done;
    auto closures = new ors::ClosureGroup<>(iters, &sync_done);
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePush("run_concurrent");
    #endif
    
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
    uint64_t start_ns = butil::cpuwide_time_ns();
    sync_done.wait();
    CHECK(sync_done.status().ok());
    // torch::cuda::synchronize(2);
    uint64_t latency_ns = butil::cpuwide_time_ns() - start_ns;
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePop();
    #endif
    std::cout << "total infer time(ns): " << latency_ns << "\n";
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
    scheduler->start();
    scoped_refptr<ors::Master> master = new ors::Master(engine, scheduler);
    std::string model_id = "llama3";
    std::map<std::string, std::string> sub_modules;
    sub_modules["submod_initial_partial"] = "/home/wangyulong/ORS/test/llama3_infer_TS/ts_fw_submods_submod_initial_partial.txt";
    for (int layer = 0; layer < 32; layer++) {
        std::ostringstream key, value;
        key << "submod_p_model_layers_" << layer << "_";
        value << "/home/wangyulong/ORS/test/llama3_infer_TS/ts_fw_submods_submod_p_model_layers_" << layer << "_.txt";
        sub_modules[key.str()] = value.str();
    }
    master->register_model(model_id, "/home/wangyulong/ORS/test/llama3_infer_TS/ts_fw_root.txt", sub_modules);

    // submit infer input
    std::vector<torch::jit::IValue> inputs;
    // get fw params
    torch::jit::Module module = torch::jit::load("/home/wangyulong/ORS/test/llama3_infer_TS/fw_params_container.pt");
    c10::IValue ivalue = module.attr("buffers");
    CHECK(ivalue.isTuple());
    for (auto element : ivalue.toTuple()->elements()) {
        inputs.push_back(element.toIValue());
    }
    // get fw buffers
    module = torch::jit::load("/home/wangyulong/ORS/test/llama3_infer_TS/fw_buffer_container.pt");
    ivalue = module.attr("buffers");
    CHECK(ivalue.isTuple());
    for (auto element : ivalue.toTuple()->elements()) {
        inputs.push_back(element.toIValue());
    }
    // get example inputs
    module = torch::jit::load("/home/wangyulong/ORS/test/llama3_infer_TS/example_inputs_container.pt");
    ivalue = module.attr("buffers");
    CHECK(ivalue.isTuple());
    for (auto element : ivalue.toTuple()->elements()) {
        inputs.push_back(element.toIValue());
    }

    c10::Device device(c10::kCUDA, 0);
    auto move_inputs = move_input_to_device(inputs, device);
    // 确保数据全部迁移到设备上了
    force_global_synchronize_debug();
    run_seq_request(master, model_id, move_inputs, 50);
    // run_seq_request(master, model_id, move_inputs, 50);
    // run_concurrent_request(master, model_id, move_inputs, 50);
    // run_concurrent_request(master, model_id, move_inputs, 50);
    force_global_synchronize_debug();

    master->stop();
    return 0;
}