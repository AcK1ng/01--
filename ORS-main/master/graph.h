#pragma once
#include <butil/memory/ref_counted.h>
#include "master/operator.h"

namespace ors {

class Model;

class Graph : public butil::RefCountedThreadSafe<Graph> {
friend class Operator;
public:
    Graph(Model* own_model, uint64_t gid,
            scoped_refptr<Operator> root,
            scoped_refptr<Operator> output,
            torch::jit::Graph* jit_graph,
            std::vector<scoped_refptr<Operator>> all_ops);
    ~Graph() {}
    void reset();
    uint64_t gid();
    scoped_refptr<Operator> root();
    butil::Status TryCopyGraphInput(std::vector<torch::jit::IValue>& inputs);
    void TryGetGraphOutput(std::vector<std::optional<torch::jit::IValue>>& outputs);
    std::optional<torch::jit::IValue> get_value(int64_t input_id);
    void register_output(int64_t output_id, torch::jit::IValue output_value);
    torch::jit::Graph* get_jit_graph();
    std::vector<scoped_refptr<Operator>> get_all_ops();
    void set_done(Closure* done);
    Closure* get_done();

    scoped_refptr<Graph> copy();

    torch::jit::IValue& reg(size_t reg) {
        return *(_registers.end() - reg);
    }
    void release_reg(size_t reg) {
        *(_registers.end() - reg) = torch::jit::IValue();
    }
public:
    Model* _own_model;
private:
    std::string _model_id;
    uint64_t _gid;
    scoped_refptr<Operator> _root;
    scoped_refptr<Operator> _output;
    torch::jit::Graph* _jit_graph;
    std::unordered_map<int64_t, torch::jit::IValue> _v_map;
    
    Closure* _done = nullptr;
    std::mutex _mutex;
public:
    std::vector<scoped_refptr<Operator>> _all_ops;
    std::vector<torch::jit::IValue> _registers;
    std::deque<std::atomic<int>> _use_count;
    std::deque<std::atomic<int>> _ref_count;
};


}
