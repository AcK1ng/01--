#include <unordered_map>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/serialization/import.h>
#include "master/model.h"
#include "master/operator.h"
#include "master/rpool.h"

namespace ors {

Model::Model(const ModelOptions& options) {
    _path = options.model_path;
    _model_name = options.model_name;
    _model_id = options.model_id;
    _sub_modules= options.sub_modules;
    _jit_graph = std::make_shared<torch::jit::Graph>();
    _global_graph_id = 0;
}

bool Model::inline_submodule(std::shared_ptr<torch::jit::Graph> main_graph, 
                      torch::jit::Node* call_node,
                      std::shared_ptr<torch::jit::Graph> submod_graph) {
    torch::jit::WithInsertPoint guard(call_node);
    std::vector<torch::jit::Value*> input_values;
    for (auto v : call_node->inputs()) {
        input_values.push_back(v);
    }
    // clone subgraph to main graph
    auto output_values = torch::jit::insertGraph(*main_graph, *submod_graph, input_values);

    if (call_node->outputs().size() == output_values.size()) {
        for (size_t i = 0; i < output_values.size(); ++i) {
            call_node->outputs()[i]->replaceAllUsesWith(output_values[i]);
        }
    } else {
        LOG(ERROR) << "call_node output not matched with submodule output size";
        return false;
    }
    call_node->destroy();
    return true;
}

bool Model::parse_subir(std::shared_ptr<torch::jit::Graph> graph, 
                        torch::jit::Node* node, 
                        std::string sub_model) {
    std::ifstream file(sub_model);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file: " << sub_model;
        return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string ir_str = buffer.str();

    std::shared_ptr<torch::jit::Graph> sub_graph = std::make_shared<torch::jit::Graph>();
    torch::jit::parseIR(ir_str, sub_graph.get());
    return inline_submodule(graph, node, sub_graph);
    // if (!inline_submodule(graph, node, sub_graph)) {
    //     LOG(ERROR) << "Failed to inline submodule: " << sub_model;
    //     return false;
    // }
    // return true;
}

void Model::split_graph_inputs(std::vector<InputSpec> inputs_spec) {
    auto original_inputs = _jit_graph->inputs().vec();
    CHECK(inputs_spec.size() == original_inputs.size());
    int weight_idx = 0, buffer_idx = 0;
    for (size_t i = 0; i < inputs_spec.size(); ++i) {
        if (inputs_spec[i].role == InputRole::ROLE_WEIGHT) {
            auto weight_val = original_inputs[i];
            torch::jit::Node* weight_node = _jit_graph->create(torch::jit::Symbol::fromQualString("ors::LoadWeight"));
            weight_node->i_(torch::jit::attr::index, weight_idx++);
            _jit_graph->prependNode(weight_node);
            weight_node->output()->setType(weight_val->type());
            weight_val->replaceAllUsesWith(weight_node->output());
            _jit_graph->eraseInput(0);
        } else if (inputs_spec[i].role == InputRole::ROLE_BUFFER) {
            auto buffer_val = original_inputs[i];
            torch::jit::Node* buffer_node = _jit_graph->create(torch::jit::Symbol::fromQualString("ors::LoadBuffer"));
            buffer_node->i_(torch::jit::attr::index, buffer_idx++);
            _jit_graph->prependNode(buffer_node);
            buffer_node->output()->setType(buffer_val->type());
            buffer_val->replaceAllUsesWith(buffer_node->output());
            _jit_graph->eraseInput(0);
        }
    }
}

bool Model::parse_ir() {
    std::ifstream file(_path);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file: " << _path;
        return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string ir_str = buffer.str();
    

    torch::jit::parseIR(ir_str, _jit_graph.get());
    // iterate node within graph
    std::vector<torch::jit::Node*> all_call_nodes;
    for (auto node : _jit_graph->nodes()) {
        std::string kind = node->kind().toQualString();
        if (kind == "prim::GCG_Call_Submod") {
            all_call_nodes.push_back(node);
        }
    }

    for (auto node : all_call_nodes) {
        if (!parse_subir(_jit_graph, node,
                _sub_modules[node->s(torch::jit::Symbol::attr("target"))])) {
            return false;
        }
    }
    if (_model_name == "llama3") {
        // for llama3, we need to split the original graph input into weight and buffer, so that we can load them separately
        std::vector<InputSpec> inputs_spec;
        // 第一个是weight,第二、三个是buffer,剩下是Input
        for (size_t i = 0; i < _jit_graph->inputs().size(); ++i) {
            if (i == 0) {
                inputs_spec.push_back({"input_" + std::to_string(i), InputRole::ROLE_WEIGHT});
            } else if (i == 1 || i == 2) {
                inputs_spec.push_back({"input_" + std::to_string(i), InputRole::ROLE_BUFFER});
            } else {
                inputs_spec.push_back({"input_" + std::to_string(i), InputRole::ROLE_INPUT});
            }
        }
        split_graph_inputs(inputs_spec);
    }
    // analyze_and_display_graph(_jit_graph);
    return true;
}
/*
1. generate graph
2. process node
*/
void Model::compile_graph() {
    std::vector<scoped_refptr<Operator>> all_ops;
    std::unordered_map<size_t, scoped_refptr<Operator>> output_to_op;
    uint64_t g_op_id = 0;
    

    // construct input and output operators
    auto input_node = _jit_graph->param_node();
    scoped_refptr<Operator> input_op_node = new Operator(input_node->kind().toQualString(),
                                                         input_node, 0, _model_id, g_op_id);
    g_op_id++;
    all_ops.push_back(input_op_node);
    // construct map from output unique id to Operator node
    for (auto output : input_node->outputs()) {
        output_to_op.insert(std::make_pair(output->unique(), input_op_node));
    }


    // construct normal operators
    for (auto node : _jit_graph->nodes()) {
        // find unresolved tensor indices
        std::vector<int64_t> unresolved_tensor_indices;
        unresolved_tensor_indices.clear();
        for (size_t i = 0; i < node->inputs().size(); i++) {
            auto input_id = node->inputs()[i]->unique(); 
            unresolved_tensor_indices.push_back(input_id);        
        }

        std::set<int64_t> dependency_indices(unresolved_tensor_indices.begin(),
                                             unresolved_tensor_indices.end());

        scoped_refptr<Operator> op_node = new Operator(node->kind().toQualString(),
                                                       node, node->inputs().size(),
                                                       _model_id, g_op_id,
                                                       unresolved_tensor_indices,
                                                       dependency_indices);
        g_op_id++;
        all_ops.push_back(op_node);
        // construct map from output unique id to Operator node
        for (auto output : node->outputs()) {
            output_to_op.insert(std::make_pair(output->unique(), op_node));
        }
    }


    
    std::vector<int64_t> unresolved_tensor_indices = {};
    auto output_node = _jit_graph->return_node();
    // find unresolved tensor indices
    for (size_t i = 0; i < output_node->inputs().size(); i++) {
        auto input_id = output_node->inputs()[i]->unique(); 
        unresolved_tensor_indices.push_back(input_id);        
    }
    std::set<int64_t> dependency_indices(unresolved_tensor_indices.begin(),
                                         unresolved_tensor_indices.end());
    scoped_refptr<Operator> output_op_node = new Operator(output_node->kind().toQualString(),
                                                          output_node, output_node->inputs().size(),
                                                          _model_id, g_op_id,
                                                          unresolved_tensor_indices,
                                                          dependency_indices);
    g_op_id++;
    all_ops.push_back(output_op_node);


    // construct parent-child relationship
    for (auto op_node : all_ops) {
        for (auto index : op_node->get_unresolved_tensor_indices()) {
            if (output_to_op.count(index)) {
                auto parent = output_to_op[index];
                parent->add_child_op(op_node.get());
                op_node->add_parent_op(parent.get());
            }
        }
    }

    // generate graph id
    uint64_t graph_id;
    {
        std::unique_lock<std::mutex> lck(_mutex);
        graph_id = _global_graph_id++;
    }

    _compile_graph = new Graph(this, graph_id, 
                        input_op_node, output_op_node,
                        _jit_graph.get(), all_ops);
    _compile_graph->_own_model = this;
    // set own_graph of operator
    for (auto op_node : all_ops) {
        op_node->set_own_graph(_compile_graph.get());
        process_node_input(op_node);
        process_node_output(op_node);
    }
    _compile_graph->_registers.resize(_compile_graph->_registers.size() + this->_register_size);
    _compile_graph->_use_count.resize(_compile_graph->_use_count.size() + this->_max_value_id + 1);

    // process_node_input()
    // load_constant_value(_compile_graph);

    // process node
    for (auto op_node : all_ops) {
        process_node(op_node);
    }
}

void Model::process_node(scoped_refptr<Operator>& node) {
    if (node->get_jit_node()->kind() == 
            torch::jit::Symbol::fromQualString("ors::LoadWeight") ||
        node->get_jit_node()->kind() == 
            torch::jit::Symbol::fromQualString("ors::LoadBuffer")) {
        return;
    }
    switch (node->get_jit_node()->kind()) {
        case torch::jit::prim::Param:
        case torch::jit::prim::Return:
        case torch::jit::prim::ListConstruct:
        case torch::jit::prim::TupleConstruct:
        case torch::jit::prim::TupleIndex:
            break;
        case torch::jit::prim::Constant:
            process_constant_node(node);
            break;
        default:
            process_operator_node(node);
            break;
    }
}

void Model::process_constant_node(scoped_refptr<Operator>& node) {
    // node->execute();
    // for (auto& child : node->_child_ops) {
    //     int32_t prev_degree = child->decrease_in_degree();
    //     CHECK(prev_degree >= 1);
    //     if (prev_degree == 1) {
    //         child->set_status(OpStatus::OP_READY);
    //         child->execute();
    //     }
    // }
    node->execute();
    node->set_status(OpStatus::OP_COMPLETED);
    propagate_static_node(node);
}

void Model::propagate_static_node(scoped_refptr<Operator> node) {
    // node->execute();
    for (auto& op : node->_child_ops) {
        int32_t prev_degree = op->decrease_in_degree();
        CHECK(prev_degree >= 1);
        if (prev_degree == 1) {
            op->execute();
            op->set_status(OpStatus::OP_COMPLETED);
            propagate_static_node(op);
        }
    }
}

void Model::process_operator_node(scoped_refptr<Operator>& op_node) {
    if (op_node->get_in_degree() == 0) {
        op_node->execute();
        op_node->set_status(OpStatus::OP_COMPLETED);
        propagate_static_node(op_node);
    }
    const torch::jit::Node* node = op_node->get_jit_node();
    const torch::jit::Operator& op = node->getOperator();
    int operator_index = add_to_operator_table(op, node);
    op_node->_operator_index = operator_index;
    op_node->_function = _operator_table[operator_index];
}

int Model::add_to_operator_table(const torch::jit::Operator& op, const torch::jit::Node* node) {
    int size = _operator_table.size();
    const torch::jit::Operation& oper = op.getOperation(node);
    _operator_table.emplace_back(oper);
    return size;
}


void Model::process_node_input(scoped_refptr<Operator>& op_node) {
    for (torch::jit::Value* input : op_node->get_jit_node()->inputs()) {
        if (input->unique() > _max_value_id) {
            _max_value_id = input->unique();
        }
        int reg = registerFor(input);
        bool moved = input->uses().size() == ++_use_count[input];
        op_node->_move_input_afer_use.push_back(moved);
        op_node->_input_reg_index.push_back(reg);
        op_node->_input_value_ids.push_back(input->unique());
        op_node->_input_uses.push_back(input->uses().size());
    }
}

void Model::process_node_output(scoped_refptr<Operator>& op_node) {
    size_t N = op_node->get_jit_node()->outputs().size();
    if (N == 0) {
        return;
    }
    int regs = allocRegs(op_node->get_jit_node()->outputs());
    for (size_t i = 0; i < op_node->get_jit_node()->outputs().size(); ++i) {
        op_node->_output_reg_index.push_back(regs + i);
    }
}

int Model::allocRegs(at::ArrayRef<torch::jit::Value*> vs) {
    int result = _register_size + 1;
    for (torch::jit::Value* v : vs) {
        CHECK(_value_to_reg.count(v) == 0);
        _value_to_reg[v] = ++_register_size;
    }
    return result;
}


/*
void Model::process_constant() {

}

void Model::process_operator() {

}

void Model::process_tupleIndex() {

}

void Model::process_
*/

scoped_refptr<Graph> Model::generate_graph() {
    return nullptr;
    // std::vector<scoped_refptr<Operator>> all_ops;
    // std::unordered_map<size_t, scoped_refptr<Operator>> output_to_op;
    // uint64_t g_op_id = 0;
    // // construct normal operators
    // for (auto node : _jit_graph->nodes()) {
    //     // find unresolved tensor indices
    //     std::vector<int64_t> unresolved_tensor_indices;
    //     unresolved_tensor_indices.clear();
    //     for (size_t i = 0; i < node->inputs().size(); i++) {
    //         auto input_id = node->inputs()[i]->unique(); 
    //         unresolved_tensor_indices.push_back(input_id);        
    //     }

    //     std::set<int64_t> dependency_indices(unresolved_tensor_indices.begin(),
    //                                          unresolved_tensor_indices.end());

    //     scoped_refptr<Operator> op_node = new Operator(node->kind().toQualString(),
    //                                                    node, node->inputs().size(),
    //                                                    _model_id, g_op_id,
    //                                                    unresolved_tensor_indices,
    //                                                    dependency_indices);
    //     g_op_id++;
    //     all_ops.push_back(op_node);
    //     // construct map from output unique id to Operator node
    //     for (auto output : node->outputs()) {
    //         output_to_op.insert(std::make_pair(output->unique(), op_node));
    //     }
    // }

    // // construct input and output operators
    // auto input_node = _jit_graph->param_node();
    // scoped_refptr<Operator> input_op_node = new Operator(input_node->kind().toQualString(),
    //                                                      input_node, 0, _model_id, g_op_id);
    // g_op_id++;
    // all_ops.push_back(input_op_node);
    // // construct map from output unique id to Operator node
    // for (auto output : input_node->outputs()) {
    //     output_to_op.insert(std::make_pair(output->unique(), input_op_node));
    // }
    // std::vector<int64_t> unresolved_tensor_indices = {};
    // auto output_node = _jit_graph->return_node();
    // // find unresolved tensor indices
    // for (size_t i = 0; i < output_node->inputs().size(); i++) {
    //     auto input_id = output_node->inputs()[i]->unique(); 
    //     unresolved_tensor_indices.push_back(input_id);        
    // }
    // std::set<int64_t> dependency_indices(unresolved_tensor_indices.begin(),
    //                                      unresolved_tensor_indices.end());
    // scoped_refptr<Operator> output_op_node = new Operator(output_node->kind().toQualString(),
    //                                                       output_node, output_node->inputs().size(),
    //                                                       _model_id, g_op_id,
    //                                                       unresolved_tensor_indices,
    //                                                       dependency_indices);
    // g_op_id++;
    // all_ops.push_back(output_op_node);


    // // construct parent-child relationship
    // for (auto op_node : all_ops) {
    //     for (auto index : op_node->get_unresolved_tensor_indices()) {
    //         if (output_to_op.count(index)) {
    //             auto parent = output_to_op[index];
    //             parent->add_child_op(op_node);
    //             op_node->add_parent_op(parent.get());
    //         }
    //     }
    // }

    // // generate graph id
    // uint64_t graph_id;
    // {
    //     std::unique_lock<std::mutex> lck(_mutex);
    //     graph_id = _global_graph_id++;
    // }
    // scoped_refptr<Graph> graph = new Graph(this, graph_id, 
    //                     input_op_node, output_op_node,
    //                     _jit_graph.get(), all_ops);

    // // set own_graph of operator
    // for (auto op_node : all_ops) {
    //     op_node->set_own_graph(graph.get());
    // } 
    // // add graph
    // add_graph(graph);
    // // load constant value
    // load_constant_value(graph);
    // // set ready state for constant node and device node
    // std::vector<scoped_refptr<Operator>> tmp_ops;
    // for (auto op : graph->get_all_ops()) {
    //     if (op->get_jit_node()->kind() == torch::jit::prim::Constant) {
    //         op->set_status(OpStatus::OP_READY);
    //         tmp_ops.push_back(op);
    //     } else if (strcmp(op->get_jit_node()->kind().toQualString(),
    //                "prim::GCG_get_native_device") == 0) {
    //         op->set_status(OpStatus::OP_READY);
    //         tmp_ops.push_back(op);
    //     }
    // }
    // g_rpool->rpool_push_operator(tmp_ops, tmp_ops.size(), 0);
    // // g_rpool->batch_push_operator(tmp_ops);

    // return graph;
}

scoped_refptr<Graph> Model::simply_generate_graph() {
    std::vector<scoped_refptr<Operator>> all_ops;
    std::unordered_map<size_t, scoped_refptr<Operator>> output_to_op;
    uint64_t g_op_id = 0;
    

    // construct input and output operators
    auto input_node = _jit_graph->param_node();
    scoped_refptr<Operator> input_op_node = new Operator(input_node->kind().toQualString(),
                                                         input_node, 0, _model_id, g_op_id);
    g_op_id++;
    all_ops.push_back(input_op_node);
    // construct map from output unique id to Operator node
    for (auto output : input_node->outputs()) {
        output_to_op.insert(std::make_pair(output->unique(), input_op_node));
    }


    // construct normal operators
    for (auto node : _jit_graph->nodes()) {
        // find unresolved tensor indices
        std::vector<int64_t> unresolved_tensor_indices;
        unresolved_tensor_indices.clear();
        for (size_t i = 0; i < node->inputs().size(); i++) {
            auto input_id = node->inputs()[i]->unique(); 
            unresolved_tensor_indices.push_back(input_id);        
        }

        std::set<int64_t> dependency_indices(unresolved_tensor_indices.begin(),
                                             unresolved_tensor_indices.end());

        scoped_refptr<Operator> op_node = new Operator(node->kind().toQualString(),
                                                       node, node->inputs().size(),
                                                       _model_id, g_op_id,
                                                       unresolved_tensor_indices,
                                                       dependency_indices);
        g_op_id++;
        all_ops.push_back(op_node);
        // construct map from output unique id to Operator node
        for (auto output : node->outputs()) {
            output_to_op.insert(std::make_pair(output->unique(), op_node));
        }
    }


    
    std::vector<int64_t> unresolved_tensor_indices = {};
    auto output_node = _jit_graph->return_node();
    // find unresolved tensor indices
    for (size_t i = 0; i < output_node->inputs().size(); i++) {
        auto input_id = output_node->inputs()[i]->unique(); 
        unresolved_tensor_indices.push_back(input_id);        
    }
    std::set<int64_t> dependency_indices(unresolved_tensor_indices.begin(),
                                         unresolved_tensor_indices.end());
    scoped_refptr<Operator> output_op_node = new Operator(output_node->kind().toQualString(),
                                                          output_node, output_node->inputs().size(),
                                                          _model_id, g_op_id,
                                                          unresolved_tensor_indices,
                                                          dependency_indices);
    g_op_id++;
    all_ops.push_back(output_op_node);


    // construct parent-child relationship
    for (auto op_node : all_ops) {
        for (auto index : op_node->get_unresolved_tensor_indices()) {
            if (output_to_op.count(index)) {
                auto parent = output_to_op[index];
                parent->add_child_op(op_node);
                op_node->add_parent_op(parent.get());
            }
        }
    }

    // generate graph id
    uint64_t graph_id;
    {
        std::unique_lock<std::mutex> lck(_mutex);
        graph_id = _global_graph_id++;
    }
    scoped_refptr<Graph> graph = new Graph(this, graph_id, 
                        input_op_node, output_op_node,
                        _jit_graph.get(), all_ops);

    // set own_graph of operator
    for (auto op_node : all_ops) {
        op_node->set_own_graph(graph.get());
    } 
    // add graph
    add_graph(graph);
    // load constant value
    load_constant_value(graph);
    // // set ready state for constant node and device node
    // std::vector<scoped_refptr<Operator>> tmp_ops;
    // for (auto op : graph->get_all_ops()) {
    //     if (op->get_jit_node()->kind() == torch::jit::prim::Constant) {
    //         op->set_status(OpStatus::OP_READY);
    //         tmp_ops.push_back(op);
    //     } else if (strcmp(op->get_jit_node()->kind().toQualString(),
    //                "prim::GCG_get_native_device") == 0) {
    //         op->set_status(OpStatus::OP_READY);
    //         tmp_ops.push_back(op);
    //     }
    // }
    // g_rpool->rpool_push_operator(tmp_ops, tmp_ops.size(), 0);
    // g_rpool->batch_push_operator(tmp_ops);

    return graph;
}

void Model::load_constant_value(scoped_refptr<Graph> graph) {
    for (auto op : graph->get_all_ops()) {
        if (op->get_jit_node()->kind() == torch::jit::prim::Constant) {
        //    if (op->get_jit_node()->output()->type()->kind() == torch::jit::FunctionType::Kind) {
        //         return;
        //     }
        //     op->register_output(op->get_jit_node()->output()->unique(), 
        //         torch::jit::toIValue(op->get_jit_node()->output()).value()); 
            op->_stack.push_back(torch::jit::toIValue(op->get_jit_node()->output()).value());
            op->extract_output(op->_stack);
        }
    }
}

void Model::load_weights(const std::string& path) {
    torch::jit::Module module = torch::jit::load(path);
    c10::IValue ivalue = module.attr("buffers");
    CHECK(ivalue.isTuple());
    for (auto element : ivalue.toTuple()->elements()) {
        _weights.push_back(element.toIValue());
    }

    // int weight_idx = 0;
    for (auto& op : _compile_graph->get_all_ops()) {
        if (op->get_jit_node()->kind() == 
                torch::jit::Symbol::fromQualString("ors::LoadWeight")) {
            // execute load weight node
            op->set_status(OpStatus::OP_COMPLETED);
            // op->_stack = _weights;
            // op->_stack.push_back(_weights[weight_idx++]);
            op->_stack.push_back(_weights[op->get_jit_node()->i(torch::jit::attr::index)]);

            op->extract_output(op->_stack);
            // continue to propagate the effect of this weight to its child node, which may also be static node
            propagate_static_node(op);
        }
    }
}

void Model::load_buffers(const std::string& path) {
    torch::jit::Module module = torch::jit::load(path);
    c10::IValue ivalue = module.attr("buffers");
    CHECK(ivalue.isTuple());
    for (auto element : ivalue.toTuple()->elements()) {
        _buffers.push_back(element.toIValue());
    }

    // int buffer_idx = 0;
    // find load buffer node and execute
    for (auto& op : _compile_graph->get_all_ops()) {
        if (op->get_jit_node()->kind() == 
                torch::jit::Symbol::fromQualString("ors::LoadBuffer")) {
            // execute load buffer node
            // op->_stack = _buffers;
            op->_stack.push_back(_buffers[op->get_jit_node()->i(torch::jit::attr::index)]);
            // op->_stack.push_back(_buffers[buffer_idx++]);
            op->extract_output(op->_stack);
            op->set_status(OpStatus::OP_COMPLETED);
            // continue to propagate the effect of this buffer to its child node, which may also be static node
            propagate_static_node(op);
        }
    }
}

void Model::analyze_and_display_graph(std::shared_ptr<torch::jit::Graph> graph) {
    std::cout << "=== TorchScript 计算图分析 ===" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // 1. 显示图的基本信息
    display_graph_basic_info(graph);
    
    // 2. 显示输入信息
    display_inputs_info(graph);
    
    // 3. 显示所有节点详细信息
    display_nodes_info(graph);
    
    // 4. 显示输出信息
    display_outputs_info(graph);
    
    // 5. 显示图统计信息
    display_graph_statistics(graph);
}

void Model::display_graph_basic_info(std::shared_ptr<torch::jit::Graph> graph) {
    std::cout << "\n1. 图基本信息:" << std::endl;
    std::cout << "   图指针: " << graph.get() << std::endl;
}

void Model::display_inputs_info(std::shared_ptr<torch::jit::Graph> graph) {
    std::cout << "\n2. 图输入分析:" << std::endl;
    auto inputs = graph->inputs();
    
    if (inputs.empty()) {
        std::cout << "   无输入" << std::endl;
        return;
    }
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];
        std::cout << "   输入 " << i << ":" << std::endl;
        std::cout << "     名称: " << input->unique() << std::endl;
        std::cout << "     类型: " << input->type()->str() << std::endl;
        std::cout << "     唯一名称: " << input->unique() << std::endl;
        
        // 显示输入用途（如果是参数或输入）
        if (input->hasUses()) {
            std::cout << "     使用次数: " << input->uses().size() << std::endl;
        }
    }
}

void Model::display_nodes_info(std::shared_ptr<torch::jit::Graph> graph) {
    std::cout << "\n3. 节点详细分析:" << std::endl;
    auto nodes = graph->nodes();
    
    int node_count = 0;
    for (const auto& node : nodes) {
        std::cout << "\n   节点 " << node_count++ << ":" << std::endl;
        display_single_node_info(node);
    }
    
    std::cout << "\n   节点 " << node_count++ << ":" << std::endl;
    display_single_node_info(graph->param_node());

    std::cout << "\n   节点 " << node_count++ << ":" << std::endl;
    display_single_node_info(graph->return_node());
}

void Model::display_single_node_info(const torch::jit::Node* node) {
    // 节点基本信息
    std::cout << "     操作类型: " << node->kind().toDisplayString() << std::endl;
    std::cout << "     完整类型: " << node->kind().toQualString() << std::endl;    
    // 显示输入
    auto inputs = node->inputs();
    if (!inputs.empty()) {
        std::cout << "     输入 (" << inputs.size() << "个):" << std::endl;
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            std::cout << "       [" << i << "] " << input->unique();
            std::cout << " : " << input->type()->str() << std::endl;
        }
    }
    
    // 显示输出
    auto outputs = node->outputs();
    if (!outputs.empty()) {
        std::cout << "     输出 (" << outputs.size() << "个):" << std::endl;
        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto& output = outputs[i];
            std::cout << "       [" << i << "] " << output->unique();
            std::cout << " : " << output->type()->str() << std::endl;
        }
    }
    
    // 显示属性
    auto attributes = node->attributeNames();
    if (!attributes.empty()) {
        std::cout << "     属性 (" << attributes.size() << "个):" << std::endl;
        for (const auto& attr_name : attributes) {
            std::cout << "       " << attr_name << ": ";
            display_node_attribute(node, attr_name);
            std::cout << std::endl;
        }
    }
    
    // 显示使用信息
    if (outputs.size() == 1 && outputs[0]->hasUses()) {
        std::cout << "     使用次数: " << outputs[0]->uses().size() << std::endl;
    }
}

void Model::display_node_attribute(const torch::jit::Node* node, const torch::jit::Symbol& attr_name) {
    try {
        // 根据属性类型显示不同的信息
        if (node->kindOf(attr_name) == torch::jit::AttributeKind::i) {
            std::cout << node->i(attr_name);
        } else if (node->kindOf(attr_name) == torch::jit::AttributeKind::f) {
            std::cout << node->f(attr_name);
        } else if (node->kindOf(attr_name) == torch::jit::AttributeKind::s) {
            std::cout << node->s(attr_name);
        } else if (node->kindOf(attr_name) == torch::jit::AttributeKind::is) {
            auto int_list = node->is(attr_name);
            std::cout << "[";
            for (size_t i = 0; i < int_list.size(); ++i) {
                std::cout << int_list[i];
                if (i < int_list.size() - 1) std::cout << ", ";
            }
            std::cout << "]";
        } else if (node->kindOf(attr_name) == torch::jit::AttributeKind::fs) {
            auto float_list = node->fs(attr_name);
            std::cout << "[";
            for (size_t i = 0; i < float_list.size(); ++i) {
                std::cout << float_list[i];
                if (i < float_list.size() - 1) std::cout << ", ";
            }
            std::cout << "]";
        } else {
            std::cout << "<" << toString(node->kindOf(attr_name)) << ">";
        }
    } catch (const std::exception& e) {
        std::cout << "<无法读取属性>";
    }
}

void Model::display_outputs_info(std::shared_ptr<torch::jit::Graph> graph) {
    std::cout << "\n4. 图输出分析:" << std::endl;
    auto outputs = graph->outputs();
    
    if (outputs.empty()) {
        std::cout << "   无输出" << std::endl;
        return;
    }
    
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& output = outputs[i];
        std::cout << "   输出 " << i << ":" << std::endl;
        std::cout << "     名称: " << output->unique() << std::endl;
        std::cout << "     类型: " << output->type()->str() << std::endl;
        std::cout << "     唯一名称: " << output->unique() << std::endl;
    }
}

void Model::display_graph_statistics(std::shared_ptr<torch::jit::Graph> graph) {
    std::cout << "\n5. 图统计信息:" << std::endl;
    
    auto nodes = graph->nodes();
    auto inputs = graph->inputs();
    auto outputs = graph->outputs();
    
    std::cout << "   节点总数: " << std::distance(nodes.begin(), nodes.end()) << std::endl;
    std::cout << "   输入数量: " << inputs.size() << std::endl;
    std::cout << "   输出数量: " << outputs.size() << std::endl;
    
    // 统计不同类型的节点
    std::unordered_map<std::string, int> node_type_count;
    for (const auto& node : nodes) {
        std::string node_type = node->kind().toDisplayString();
        node_type_count[node_type]++;
    }
    
    std::cout << "   节点类型分布:" << std::endl;
    for (const auto& [type, count] : node_type_count) {
        std::cout << "     " << std::setw(20) << std::left << type 
                  << ": " << count << " 个" << std::endl;
    }
}

}