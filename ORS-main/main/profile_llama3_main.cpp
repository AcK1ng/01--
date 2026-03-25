#include <torch/script.h>
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
    for (int dev_idx = 0; dev_idx < 2; ++dev_idx) {
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
        force_global_synchronize_debug();
        ors::SynchronizedClosure done;
        ret = master->submit_infer_request_with_graph(graphs[i], move_inputs, &done);
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return;
        }
        done.wait();
        force_global_synchronize_debug();
        graphs[i]->reset();
        #ifdef ENABLE_NVTX_DEBUG
        nvtxRangePop();  // 结束当前迭代的范围
        #endif
    }
    #ifdef ENABLE_NVTX_DEBUG
    nvtxRangePop();
    #endif
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
    ors::IScheduler* scheduler = new ors::ProfileScheduler(engine);
    scheduler->start();
    scoped_refptr<ors::Master> master = new ors::Master(engine, scheduler);
    // register model
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

    // warm up in gpu
    for (int i = 0; i < 2; i++) {
        // warm up
        c10::Device device(c10::kCUDA, i);
        static_cast<ors::ProfileScheduler*>(scheduler)->set_profile_worker_id(i);
        warmup(master, model_id, inputs, 20, device);
    }

    force_global_synchronize_debug();
    ors::g_latency_est.ClearProfileData();
    for (int i = 0; i < 2; i++) {
        // profiling data in gpu
        butil::Status ret;
        scoped_refptr<ors::Graph> graph;
        c10::Device device(c10::kCUDA, i);
        static_cast<ors::ProfileScheduler*>(scheduler)->set_profile_worker_id(i);
        ret = master->prepare_graph(model_id, graph);
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return 0;
        }
            
        std::vector<torch::jit::IValue> move_inputs;
        for (auto input : inputs) {
            move_inputs.push_back(moveIValueToDevice(input, device));
        }

        #ifdef ENABLE_NVTX_DEBUG
        nvtxRangePush("ors_profile_data");
        #endif

        force_global_synchronize_debug();
        ors::SynchronizedClosure done;
        ret = master->submit_infer_request_with_graph(graph, move_inputs, &done);
        if (!ret.ok()) {
            LOG(ERROR) << "prepare graph failed, ret: " << ret;
            return 0;
        }
        done.wait();
        force_global_synchronize_debug();
        graph->reset();
        #ifdef ENABLE_NVTX_DEBUG
        nvtxRangePop();
        #endif
    }

    ors::g_latency_est.DumpProfile();

    master->stop();
    return 0;
}