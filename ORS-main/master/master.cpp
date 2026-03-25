#include "master/master.h"
#include "master/model.h"
#include "common/utils.h"
#include "master/model_manager.h"
#include "master/rpool.h"
#include "common/timing.h"
#include "master/latency_estimate.h"

namespace ors {

LatencyEstimator g_latency_est;

void Master::start() {
    _engine->start();
    _scheduler->start();
}

void Master::stop() {
    _scheduler->stop();
    _engine->stop();
}

butil::Status Master::register_model(std::string model_name, 
                                     std::string model_path, 
                                     std::map<std::string, std::string> sub_models) {
    ModelOptions options;
    options.model_path = model_path;
    options.model_name = model_name;
    options.model_id = model_name;
    options.sub_modules = sub_models;
    scoped_refptr<Model> model = new Model(options);
    if (!model->parse_ir()) {
        return butil::Status(EIO, "parse ir failed");
    }
    model->compile_graph();
    g_model_manager->add_model(model);
    return butil::Status::OK();
}

butil::Status Master::prepare_graph(std::string model_id,
                            scoped_refptr<Graph>& graph) {
    scoped_refptr<Model> model = g_model_manager->get_model(model_id);
    if (!model) {
        LOG(WARNING) << "model is not exist, model id: " << model_id;
        return butil::Status(EIO, "no such model");
    }
    graph = model->simply_generate_graph();
    if (!graph) {
        LOG(WARNING) << "Generating graph failed, model id: " << model_id;
        return butil::Status(EIO, "generating graph failed");
    }
    return butil::Status::OK();
}


butil::Status Master::get_execute_graph(std::string model_id,
                            scoped_refptr<Graph>& graph) {
    scoped_refptr<Model> model = g_model_manager->get_model(model_id);
    if (!model) {
        LOG(WARNING) << "model is not exist, model id: " << model_id;
        return butil::Status(EIO, "no such model");
    }
    graph = model->_compile_graph->copy();
    if (!graph) {
        LOG(WARNING) << "Generating graph failed, model id: " << model_id;
        return butil::Status(EIO, "generating graph failed");
    }
    return butil::Status::OK();
}

butil::Status Master::upload_tensor(std::string model_id, 
                                    int input_pos, 
                                    torch::jit::IValue input) {
    scoped_refptr<Model> model = g_model_manager->get_model(model_id);
    if (!model) {
        LOG(WARNING) << "model is not exist, model id: " << model_id;
        return butil::Status(EIO, "no such model");
    }
    auto input_node = model->get_jit_graph()->param_node();
    const auto& model_inputs = input_node->outputs();
    model->register_variables(model_inputs[input_pos]->unique(), input);
    return butil::Status::OK();
}

butil::Status Master::submit_infer_request(std::string model_id, 
    std::vector<torch::jit::IValue> inputs, Closure* done,
    uint64_t& gid) {
    // scoped_refptr<Model> model = g_model_manager->get_model(model_id);
    // if (!model) {
    //     LOG(WARNING) << "model is not exist, model id: " << model_id;
    //     return butil::Status(EIO, "no such model");
    // }
    // #ifdef ENABLE_TIME_DEBUG
    // g_generate_graph.Start();
    // #endif
    // scoped_refptr<Graph> graph = model->generate_graph();
    // if (!graph) {
    //     LOG(WARNING) << "Generating graph failed, model id: " << model_id;
    //     return butil::Status(EIO, "generating graph failed");
    // }
    // auto ret = graph->TryCopyGraphInput(inputs);
    // if (!ret.ok()) {
    //     LOG(ERROR) << "copy input failed, model id: "  << model_id;
    //     return butil::Status(EIO, "copy input failed");
    // }
    // graph->root()->set_status(OpStatus::OP_READY);
    // graph->set_done(done);
    // std::vector<scoped_refptr<Operator>> ops = {graph->root()};
    // g_rpool->rpool_push_operator(ops, 1, 0);
    // gid = graph->gid();
    // #ifdef ENABLE_TIME_DEBUG
    // g_generate_graph.Stop();
    // #endif
    // return butil::Status::OK();
    return butil::Status::OK();
}

butil::Status Master::submit_infer_request_with_graph(scoped_refptr<Graph> graph, 
    std::vector<torch::jit::IValue> inputs, Closure* done) {
    // #ifdef ENABLE_TIME_DEBUG
    // g_copy_graph_input.Start();
    // #endif
    // auto ret = graph->TryCopyGraphInput(inputs);
    // if (!ret.ok()) {
    //     LOG(ERROR) << "copy input failed, graph id: "  << graph->gid();
    //     return butil::Status(EIO, "copy input failed");
    // }
    // graph->root()->set_status(OpStatus::OP_READY);
    // graph->set_done(done);
    // std::vector<scoped_refptr<Operator>> ops = {graph->root()};
    // g_rpool->rpool_push_operator(ops, 1, 0);
    // #ifdef ENABLE_TIME_DEBUG
    // g_copy_graph_input.Stop();
    // #endif
    // return butil::Status::OK();
    return butil::Status::OK();
}

butil::Status Master::fill_graph_with_inputs(scoped_refptr<Graph>& graph,
        std::vector<torch::jit::IValue>& inputs) {
    auto ret = graph->TryCopyGraphInput(inputs);
    if (!ret.ok()) {
        LOG(ERROR) << "copy input failed, graph id: "  << graph->gid();
        return butil::Status(EIO, "copy input failed");
    }
    return butil::Status::OK();
}

butil::Status Master::run_graph(scoped_refptr<Graph>& graph, Closure* done) {
    // graph->root()->set_status(OpStatus::OP_READY);
    graph->set_done(done);
    // std::vector<scoped_refptr<Operator>> ops = {graph->root()};
    std::vector<Operator*> ops = {graph->root().get()};
    g_rpool->rpool_push_operator(ops, 1, 0);

    // std::vector<scoped_refptr<Operator>> ready_ops;
    // for (auto& op : graph->get_all_ops()) {
    //     if (op->status() == OpStatus::OP_COMPLETED) {
    //         continue;
    //     }
    //     ready_ops.emplace_back(op);   
    // }
    // g_rpool->rpool_push_operator(ready_ops, ready_ops.size(), 0);

    return butil::Status::OK();
}

void Master::drop_graph(std::string model_id, uint64_t gid) {
    scoped_refptr<Model> model = g_model_manager->get_model(model_id);
    if (!model) {
        LOG(WARNING) << "model is not exist, model id: " << model_id;
        return;
    }
    model->drop_graph(gid);
}

butil::Status Master::get_infer_result(std::string model_id, uint64_t gid,
        std::vector<std::optional<torch::jit::IValue>>& outputs) {
    scoped_refptr<Model> model = g_model_manager->get_model(model_id);
    if (!model) {
        LOG(WARNING) << "model is not exist, model id: " << model_id;
        return butil::Status(EIO, "no such model");
    } else {
        scoped_refptr<Graph> graph = model->get_graph(gid);
        if (!graph) {
            LOG(WARNING) << "get infer result failed, model id: " << model_id
                            << ", graph id: " << gid;
            return butil::Status(EIO, "get infer result failed");
        }
        graph->TryGetGraphOutput(outputs);
    }
    return butil::Status::OK();
}

butil::Status Master::get_graph_output(scoped_refptr<Graph> graph, 
        std::vector<std::optional<torch::jit::IValue>>& outputs) {
    graph->TryGetGraphOutput(outputs);
    return butil::Status::OK();
}

butil::Status Master::load_weights(const std::string& model_id,
                               const std::string& path) {
    scoped_refptr<Model> model = g_model_manager->get_model(model_id);
    if (!model) {
        LOG(WARNING) << "model is not exist, model id: " << model_id;
        return butil::Status(EIO, "no such model");
    }
    model->load_weights(path);
    return butil::Status::OK();
}

butil::Status Master::load_buffers(const std::string& model_id,
                               const std::string& path) {
    scoped_refptr<Model> model = g_model_manager->get_model(model_id);
    if (!model) {
        LOG(WARNING) << "model is not exist, model id: " << model_id;
        return butil::Status(EIO, "no such model");
    }
    model->load_buffers(path);
    return butil::Status::OK();
}

} // namespace ors