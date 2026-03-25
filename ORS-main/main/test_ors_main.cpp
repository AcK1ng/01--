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
    force_global_synchronize_debug();
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
    // sleep(60);
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
    
    // register model
    std::string model_id = "testModel";
    std::map<std::string, std::string> sub_modules;
    sub_modules["submod_fw0"] = "/home/wangyulong/ORS/GCG/UserProgram/test_module/ts_fw_submods_submod_fw0.txt";
    sub_modules["submod_fw1"] = "/home/wangyulong/ORS/GCG/UserProgram/test_module/ts_fw_submods_submod_fw1.txt";
    sub_modules["submod_fw2"] = "/home/wangyulong/ORS/GCG/UserProgram/test_module/ts_fw_submods_submod_fw2.txt";
    master->register_model(model_id, "/home/wangyulong/ORS/GCG/UserProgram/test_module/ts_fw_root.txt", sub_modules);

    // submit infer input
    std::vector<torch::jit::IValue> inputs;
    // get fw params
    torch::jit::Module module = torch::jit::load("/home/wangyulong/ORS/GCG/UserProgram/test_module/fw_params_container.pt");
    c10::IValue ivalue = module.attr("buffers");
    CHECK(ivalue.isTuple());
    for (auto element : ivalue.toTuple()->elements()) {
        inputs.push_back(element.toIValue());
    }
    // get fw buffers
    module = torch::jit::load("/home/wangyulong/ORS/GCG/UserProgram/test_module/fw_buffer_container.pt");
    ivalue = module.attr("buffers");
    CHECK(ivalue.isTuple());
    for (auto element : ivalue.toTuple()->elements()) {
        inputs.push_back(element.toIValue());
    }
    // get example inputs
    module = torch::jit::load("/home/wangyulong/ORS/GCG/UserProgram/test_module/example_inputs_container.pt");
    ivalue = module.attr("buffers");
    CHECK(ivalue.isTuple());
    for (auto element : ivalue.toTuple()->elements()) {
        inputs.push_back(element.toIValue());
    }

    // 确保数据全部迁移到设备上了
    force_global_synchronize_debug();
    // run_seq_request(master, model_id, move_inputs, 1);
    // run_seq_request(master, model_id, move_inputs, 50);
    // run_concurrent_request(master, model_id, move_inputs,100);
    // run_concurrent_request(master, model_id, move_inputs,10);
    // run_concurrent_request(master, model_id, move_inputs,10);
    for (int i = 0; i < 10; i++) {
        run_seq_request(master, model_id, inputs, 10);
        // run_seq_request(master, model_id, move_inputs, 10);
    }

    master->stop();
    return 0;
}