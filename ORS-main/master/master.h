#pragma once
#include <torch/csrc/api/include/torch/types.h>
#include <butil/memory/ref_counted.h>
#include <butil/status.h>
#include "master/rpool.h"
#include "worker/engine.h"
#include "master/scheduler.h"

namespace ors {

class Master : public butil::RefCountedThreadSafe<Master> {
public:
    Master(Engine* engine, IScheduler* scheduler) : 
        _engine(engine), 
        _scheduler(scheduler) {}
    ~Master() {}

    void start();
    void stop();

    butil::Status register_model(std::string model_name,
                                 std::string model_path,  
                                 std::map<std::string, std::string> sub_models = {});

    butil::Status prepare_graph(std::string model_id,
                                scoped_refptr<Graph>& graph);

    butil::Status get_execute_graph(std::string model_id,
                            scoped_refptr<Graph>& graph);

    butil::Status upload_tensor(std::string model_id, 
                                    int input_pos, 
                                    torch::jit::IValue input);

    butil::Status submit_infer_request(std::string model_id, 
                                       std::vector<torch::jit::IValue> inputs,
                                       Closure* done,
                                       uint64_t& gid);

    butil::Status submit_infer_request_with_graph(scoped_refptr<Graph> graph, 
                                                  std::vector<torch::jit::IValue> inputs,
                                                  Closure* done);

    butil::Status fill_graph_with_inputs(scoped_refptr<Graph>& graph,
                                         std::vector<torch::jit::IValue>& inputs);

    butil::Status run_graph(scoped_refptr<Graph>& graph, Closure* done);

    butil::Status get_infer_result(std::string model_id, uint64_t gid,
        std::vector<std::optional<torch::jit::IValue>>& outputs);

    void drop_graph(std::string model_id, uint64_t gid);

    butil::Status get_graph_output(scoped_refptr<Graph> graph,
        std::vector<std::optional<torch::jit::IValue>>& outputs);

    butil::Status load_weights(const std::string& model_id,
                               const std::string& path);

    butil::Status load_buffers(const std::string& model_id,
                               const std::string& path);

private:
    Engine* _engine;
    IScheduler* _scheduler;
};

} // namespace ors