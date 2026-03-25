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

#include <torch/csrc/jit/runtime/graph_executor.h>

namespace ors {
extern LatencyEstimator g_latency_est;
}

DECLARE_string(flagfile);


void display_opt_graph(scoped_refptr<ors::Master> master, std::string model_id) {
    // std::deque<scoped_refptr<ors::Graph>> graphs;
    scoped_refptr<ors::Graph> graph;
    master->prepare_graph(model_id, graph);
    std::cout << "graph:\n" << *(graph->get_jit_graph());

    // auto opt_graph = graph->get_jit_graph()->copy();
    // Inline(*opt_graph);
    // LowerGradOf(*opt_graph);
    // specializeAutogradZero(opt_graph);
    // LowerSimpleTuples(opt_graph);
    // ConstantPooling(opt_graph);
    // runRequiredPasses(opt_graph);
    // ConstantPropagation(opt_graph);
    // PropagateInputShapes(opt_graph);
    // PropagateRequiresGrad(opt_graph);
    // runOptimization(opt_graph);
    // EliminateDeadCode(opt_graph);
    std::shared_ptr<torch::jit::Graph> code_graph(graph->get_jit_graph());
    auto code_impl = torch::jit::Code(code_graph, "<on-demand-func>");
    std::cout << "graph:\n" << *(code_impl.graph());
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

    ors::IScheduler* scheduler = new ors::RoundRobinScheduler(engine);
    // scheduler->start();
    scoped_refptr<ors::Master> master = new ors::Master(engine, scheduler);
    std::string model_id = "llama3";
    std::map<std::string, std::string> sub_modules;
    sub_modules["submod_op"] = "/home/wangyulong/ORS/GCG/UserProgram/llama3_infer_TS/ts_fw_submods_submod_op.txt";
    master->register_model(model_id, "/home/wangyulong/ORS/GCG/UserProgram/llama3_infer_TS/ts_fw_root.txt", sub_modules);
    display_opt_graph(master, "llama3");
    master->stop();
    return 0;
}