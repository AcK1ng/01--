#pragma once
#include <string>
#include <torch/csrc/jit/ir/ir.h>
#include <master/graph.h>
#include <butil/memory/ref_counted.h>

namespace ors {

// 最开始的输入是一个weight+buffer+input的混合输入，我们需要把它们区分开来，分别处理
enum InputRole {
    ROLE_WEIGHT,
    ROLE_BUFFER,
    ROLE_INPUT
};

struct InputSpec {
    std::string name;
    InputRole role;
};

struct ModelOptions {
    std::string model_path;
    std::string model_name;
    std::string model_id;
    std::map<std::string, std::string> sub_modules = {};
};

class Model : public butil::RefCountedThreadSafe<Model> {
public:
    Model(const ModelOptions& options);
    ~Model() {}

    std::string Id() {
        return _model_id;
    }

    bool inline_submodule(std::shared_ptr<torch::jit::Graph> main_graph, 
                      torch::jit::Node* call_node,
                      std::shared_ptr<torch::jit::Graph> submod_graph);
    bool parse_subir(std::shared_ptr<torch::jit::Graph> graph,
                    torch::jit::Node* node, 
                    std::string sub_model);

    bool parse_ir();
    void split_graph_inputs(std::vector<InputSpec> inputs_spec);
    void compile_graph();
    void process_node(scoped_refptr<Operator>& node);
    void process_operator_node(scoped_refptr<Operator>& op_node);
    void process_constant_node(scoped_refptr<Operator>& node);
    void propagate_static_node(scoped_refptr<Operator> node);
    int registerFor(torch::jit::Value* v) {
        return _value_to_reg.at(v);
    }
    int allocRegs(at::ArrayRef<torch::jit::Value*> vs);
    void process_node_input(scoped_refptr<Operator>& op_node);
    void process_node_output(scoped_refptr<Operator>& op_node);

    int add_to_operator_table(const torch::jit::Operator& op,
                              const torch::jit::Node* node);
    scoped_refptr<Graph> generate_graph();
    scoped_refptr<Graph> simply_generate_graph();

    // graph manage
    void add_graph(scoped_refptr<Graph> graph) {
        std::unique_lock<std::mutex> lock(_mutex);
        _graph_map[graph->gid()] = graph;
    }
    scoped_refptr<Graph> get_graph(uint64_t gid) {
        std::unique_lock<std::mutex> lock(_mutex);
        return _graph_map[gid];
    }
    scoped_refptr<Graph> drop_graph(uint64_t gid) {
        std::unique_lock<std::mutex> lock(_mutex);
        auto it = _graph_map.find(gid);
        if (it != _graph_map.end()) {
            auto tmp_graph = _graph_map[gid];
            _graph_map.erase(it);
            return tmp_graph;
        }
        return nullptr;
    }
    
    void load_constant_value(scoped_refptr<Graph> graph);

    void load_weights(const std::string& path);
    void load_buffers(const std::string& path);

    void register_variables(int64_t id, torch::jit::IValue& var) {
        std::unique_lock<std::mutex> lock(_mutex);
        _variables[id] = var;
    }
    std::shared_ptr<torch::jit::Graph> get_jit_graph() {
        return _jit_graph;
    }
    
    std::vector<torch::jit::Operation> _operator_table;
    scoped_refptr<Graph> _compile_graph;
    std::unordered_map<torch::jit::Value*, size_t> _use_count;
    int _register_size = 0;
    std::vector<torch::jit::IValue> _weights;
    std::vector<torch::jit::IValue> _buffers;

    size_t _max_value_id = 0;
private:
    // analyze graph info
    void analyze_and_display_graph(std::shared_ptr<torch::jit::Graph> graph);
    void display_graph_basic_info(std::shared_ptr<torch::jit::Graph> graph);
    void display_inputs_info(std::shared_ptr<torch::jit::Graph> graph);
    void display_nodes_info(std::shared_ptr<torch::jit::Graph> graph);
    void display_single_node_info(const torch::jit::Node* node);
    void display_node_attribute(const torch::jit::Node* node, const torch::jit::Symbol& attr_name);
    void display_outputs_info(std::shared_ptr<torch::jit::Graph> graph);
    void display_graph_statistics(std::shared_ptr<torch::jit::Graph> graph);

    uint64_t _global_graph_id;
    std::string _path;
    std::map<std::string, std::string> _sub_modules;
    std::string _model_name;
    std::string _model_id;
    std::map<uint32_t, scoped_refptr<Graph>> _graph_map;
    std::shared_ptr<torch::jit::Graph> _jit_graph;
    // value to save buffer and params, unique_id----->Ivalue
    std::unordered_map<int64_t, torch::jit::IValue> _variables;

    scoped_refptr<Graph> _graph;

    std::unordered_map<torch::jit::Value*, int> _value_to_reg;

    std::mutex _mutex;
};

} // namespace ors