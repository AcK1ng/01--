#include "master/graph.h"
#include "master/model.h"

namespace ors {

Graph::Graph(Model* own_model, uint64_t gid,
            scoped_refptr<Operator> root,
            scoped_refptr<Operator> output,
            torch::jit::Graph* jit_graph,
            std::vector<scoped_refptr<Operator>> all_ops) : 
        _own_model(own_model), _gid(gid), 
        _root(root), _output(output),
        _jit_graph(jit_graph), _all_ops(all_ops) {
    // _registers.resize(_registers.size() + _own_model->_register_size);
    // _use_count.resize(_use_count.size() + _jit_graph->);
}

void Graph::reset() {
    _done = nullptr;
    for (auto& op : _all_ops) {
        op->reset();
    }
}

butil::Status Graph::TryCopyGraphInput(std::vector<torch::jit::IValue>& inputs) {
    // const auto& graph_inputs = _root->get_jit_node()->outputs();
    // size_t start_pos = _root->_input_map.size();
    // for (int i = 0; start_pos < graph_inputs.size(); ++start_pos, ++i) {
    //     torch::jit::Value* val = graph_inputs[start_pos];
    //     torch::jit::IValue iv = inputs[i];
    //     _root->register_output(val->unique(), iv);
    // }
    CHECK(inputs.size() == _root->get_jit_node()->outputs().size());
    _root->_stack = inputs;
    _root->extract_output(_root->_stack);
    return butil::Status::OK();
}

void Graph::TryGetGraphOutput(std::vector<std::optional<torch::jit::IValue>>& outputs) {
    std::unique_lock<std::mutex> lck(_mutex);
    for (auto output : _output->get_jit_node()->inputs()) {
        auto it = _output->_input_map.find(output->unique());
        if (it != _output->_input_map.end()) {
            outputs.push_back(it->second);
        } else {
            outputs.push_back(std::nullopt);
        }
    }
}

std::optional<torch::jit::IValue> Graph::get_value(int64_t input_id) {
    std::unique_lock<std::mutex> lck(_mutex);
    auto it = _v_map.find(input_id);
    if (it == _v_map.end()) {
        return std::nullopt;
    }
    return it->second;
}

void Graph::set_done(Closure* done) {
    _done = done;
}

Closure* Graph::get_done() {
    return _done;
}

void Graph::register_output(int64_t output_id, torch::jit::IValue output_value) {
    std::unique_lock<std::mutex> lck(_mutex);
    _v_map[output_id] = output_value;
}

torch::jit::Graph* Graph::get_jit_graph() {
    return _jit_graph;
}

std::vector<scoped_refptr<Operator>> Graph::get_all_ops() {
    return _all_ops;
}

scoped_refptr<Operator> Graph::root() { 
    return _root; 
}

uint64_t Graph::gid() {
    return _gid; 
}

scoped_refptr<Graph> Graph::copy() {
    // 1. 准备映射表：Old Operator Pointer -> New Operator Pointer
    std::unordered_map<Operator*, Operator*> op_map;
    std::vector<scoped_refptr<Operator>> new_all_ops;

    // 预分配空间，避免 realloc
    new_all_ops.reserve(_all_ops.size());

    // 2. 创建新 Graph 的壳子
    // 注意：此时 root 和 output 还是空的，稍后填
    auto new_graph = new Graph(
        _own_model, _gid, 
        nullptr, nullptr, 
        _jit_graph, 
        {}
    );
    new_graph->_own_model = this->_own_model; // 如果需要反向指针

    // 3. 第一轮遍历：克隆所有节点（Clone Nodes）
    for (const auto& old_op : _all_ops) {
        // 调用 Operator::clone，传入新图指针
        auto new_op = old_op->clone();
        new_op->_own_graph = new_graph;

        // 存入列表和映射表
        new_all_ops.push_back(new_op);
        op_map[old_op.get()] = new_op;
    }

    // 4. 第二轮遍历：重建拓扑关系（Rebuild Topology）
    for (const auto& old_op : _all_ops) {
        auto new_op = op_map[old_op.get()];

        // 4.1 重建 _child_ops
        for (const auto& old_child : old_op->_child_ops) {
            // 通过旧指针找到对应的新指针
            if (op_map.find(old_child) != op_map.end()) {
                new_op->_child_ops.push_back(op_map[old_child]);
            }
        }

        // 4.2 重建 _parent_ops
        for (const auto& old_parent : old_op->_parent_ops) {
            if (op_map.find(old_parent) != op_map.end()) {
                // 注意 _parent_ops 存储的是裸指针
                new_op->_parent_ops.push_back(op_map[old_parent]);
            }
        }
    }

    // 5. 设置新 Graph 的关键指针
    new_graph->_all_ops = std::move(new_all_ops);
    
    if (_root && op_map.count(_root.get())) {
        new_graph->_root = op_map[_root.get()];
    }
    if (_output && op_map.count(_output.get())) {
        new_graph->_output = op_map[_output.get()];
    }
    // 6. 拷贝预分配的寄存器(里面可能提前保存了一些常量和张量)
    new_graph->_registers = this->_registers;
    // new_graph->_use_count = this->_use_count;

    
    new_graph->_use_count.clear();
    for (const auto& val : this->_use_count) {
        new_graph->_use_count.emplace_back(0);
    }

    new_graph->_ref_count.clear();
    for (const auto& val: new_graph->_use_count) {
        new_graph->_ref_count.emplace_back(val.load(std::memory_order_relaxed));
    }

    // new_graph->_use_count.clear();
    // // new_graph->_use_count.reserve(this->_use_count.size());
    // for (size_t i = 0; i < this->_use_count.size(); ++i) {
    //     new_graph->_use_count.push_back(
    //         this->_use_count[i].load(std::memory_order_relaxed)
    //     );
    // }
    return new_graph;
}


}