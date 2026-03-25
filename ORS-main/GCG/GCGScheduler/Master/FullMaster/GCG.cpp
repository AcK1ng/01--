#include "GCG.h"
#include "Master/FullMaster/Scheduling.h"
#include "Cluster/OOQueue/OOQueue.h"

#include <exprtk.hpp>

#include <torch/jit.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/interpreter/code_impl.h>

static std::shared_ptr<VariableDescriptor>
torch_jit_type__to__VariableDescriptor(c10::TypePtr jit_type) {
    std::shared_ptr<VariableDescriptor> ret = nullptr;

    if (jit_type->kind() == c10::TypeKind::TensorType) {
        ret = std::make_shared<VariableDescriptor>();
        auto type = jit_type->cast<c10::TensorType>();
        ret->is_tuple = false;
        ret->is_tensor = true;
        ret->shape.reserve(type->dim().value());
        for (size_t dim = 0; dim < type->dim(); dim++)
            ret->shape.push_back(type->symbolic_sizes()[dim].value());
        ret->dtype = (int)type->scalarType().value();
    } else if (jit_type->kind() == c10::TypeKind::TupleType) {
        ret = std::make_shared<VariableDescriptor>();
        auto type = jit_type->castRaw<c10::TupleType>();
        ret->is_tuple = true;
        ret->is_tensor = false;
        ret->elems.reserve(type->elements().size());
        for (size_t i = 0; i < type->elements().size(); i++)
            ret->elems.push_back(torch_jit_type__to__VariableDescriptor(type->elements().at(i)));
    } else if (jit_type->kind() == c10::TypeKind::NoneType) {
    } else if (jit_type->kind() == c10::TypeKind::FloatType
        || jit_type->kind() == c10::TypeKind::IntType) {
        ret = std::make_shared<VariableDescriptor>();
        ret->is_tuple = false;
        ret->is_tensor = false;
    } else 
        assert(false);

    return ret;
}

Graph
GCG_Adding_TaskManagement::GetGraphFrom(std::string ts_graph_str) {
    torch::jit::Graph TS_graph = torch::jit::Graph();
    torch::jit::parseIR(ts_graph_str, &TS_graph);

    Graph ret;
    std::unordered_map<size_t, NodeID> ts_node__to__node_id;
    for (auto TS_input_value: TS_graph.inputs()) {
        auto *input = ret.__add_node({});
        input->_.shape = torch_jit_type__to__VariableDescriptor(TS_input_value->type());
        ts_node__to__node_id[TS_input_value->unique()] = input->node_id;
        ret.inputs.push_back(input->node_id);
    }

    for (auto TS_node: TS_graph.nodes()) {
        std::vector<NodeID> node_inputs;
        for (auto TS_input_value: TS_node->inputs())
            node_inputs.push_back(ts_node__to__node_id[TS_input_value->unique()]);
        struct Node *node = ret.__add_node(node_inputs);
        auto node_value = TS_node->output();
        std::optional<torch::jit::IValue> _const_ival = torch::jit::toIValue(node_value);
        node->_.lost_inputs = false;
        node->_.has_tensor_payload = false;
        if (_const_ival.has_value()) {
            torch::jit::IValue const_ival = _const_ival.value();
            node->_.is_constant = true;
            node->_.const_payload = MyConstantPayload(const_ival);
        } else {
            node->_.is_constant = false;
            std::string kind = TS_node->kind().toQualString();
            if (kind == "prim::GCG_Call_Submod")
                node->_.target = TS_node->s(torch::jit::Symbol::attr("target"));
            else
                node->_.aten_op_name = kind;
            node->_.shape = torch_jit_type__to__VariableDescriptor(node_value->type());
        }
        assert(TS_node->outputs().size() == 1);
        ts_node__to__node_id[TS_node->output(0)->unique()] = node->node_id;
    }
    for (auto TS_output_value: TS_graph.outputs())
        ret.outputs.push_back(ts_node__to__node_id[TS_output_value->unique()]);
    return ret;
}



void
GCG_Adding_CheckpointManagement::__init_cached_symbol_value__for_shape_propagation(
    std::vector<NodeID> inputs,
    std::shared_ptr<Graph> root,
    std::unordered_map<std::string, size_t> &cached_symbol_id_value,
    std::unordered_map<int, size_t> &cached_symbol_value,
    const std::unordered_map<int, std::string> &symbol__to__symexpr) {
    assert(inputs.size() == root->inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        NodeID input_id = inputs[i];
        auto *input = this->__get_node(input_id);
        std::shared_ptr<VariableDescriptor> complete_shape = input->_.shape;
        std::shared_ptr<VariableDescriptor> incomplete_shape = root->__get_node(root->inputs[i])->_.shape;
        for_each_leaf__of_two_tuple_tree(complete_shape, incomplete_shape,
            [&] (std::shared_ptr<VariableDescriptor> complete_shape,
                 std::shared_ptr<VariableDescriptor> incomplete_shape) {
                if (complete_shape->is_tensor) {
                    for (size_t dim = 0; dim < complete_shape->shape.size(); dim++) {
                        if (incomplete_shape->shape[dim] < 0) {
                            cached_symbol_id_value[
                                symbol__to__symexpr.at(incomplete_shape->shape[dim])] = complete_shape->shape[dim];
                            cached_symbol_value[incomplete_shape->shape[dim]] = complete_shape->shape[dim];
                        }
                    }
                }
            });
    }
}

struct TrueDiv final : public exprtk::ifunction<double> {
    TrueDiv() : exprtk::ifunction<double>(2) { exprtk::disable_has_side_effects(*this); }
    inline double operator()(const double& v1, const double& v2) override {
        return v1 / v2; 
    }
};

struct CeilToInt final : public exprtk::ifunction<double> {
    CeilToInt() : exprtk::ifunction<double>(1) { exprtk::disable_has_side_effects(*this); }
    inline double operator()(const double& v) override {
        return static_cast<int64_t>(std::ceil(v));
    }
};

void
GCG_Adding_CheckpointManagement::__propagate_shape__from_cached_symbol_value(
    std::shared_ptr<VariableDescriptor> incomplete_shape,
    const std::unordered_map<std::string, size_t> &cached_symbol_id_value,
    std::unordered_map<int, size_t> &cached_symbol_value,
    const std::unordered_map<int, std::string> &symbol__to__symexpr) {
    for_each_elem__in_tuple_vd(incomplete_shape,
        [&] (std::shared_ptr<VariableDescriptor> vd) {
        if (!vd->is_tensor)
            return;
        vd->numel = 1;
        for (size_t dim = 0; dim < vd->shape.size(); dim++) {
            if (vd->shape[dim] < 0) {
                int symbol = vd->shape[dim];
                if (!cached_symbol_value.contains(symbol)) {
                    exprtk::symbol_table<double> symbol_table;
                    exprtk::expression<double> expression;
                    exprtk::parser<double> parser;
                    TrueDiv true_div;
                    CeilToInt ceil_int;
                    symbol_table.add_function("IntTrueDiv", true_div);
                    symbol_table.add_function("CeilToInt", ceil_int);
                    for (const auto &[id, value]: cached_symbol_id_value)
                        symbol_table.add_constant(id, value);
                    expression.register_symbol_table(symbol_table);
                    parser.compile(symbol__to__symexpr.at(symbol), expression);
                    cached_symbol_value[symbol] = static_cast<size_t>(expression.value());
                }
                vd->shape[dim] = cached_symbol_value.at(symbol);
            }
            vd->numel *= vd->shape[dim];
        }
    });
}


std::unordered_map<NodeID, Rank>
Simple_Full_GCG::_Step_3_Op_Scheduling(
    std::function<Duration (AccModel, TaskID, TaskNodeID)> _task_predictor,
    std::function<std::pair<Duration, NBytes> (Rank, Rank, std::shared_ptr<VariableDescriptor>)> _transmit_predictor,
    std::shared_ptr<GCG_Adding_OpManagement> example_GCG,
    std::shared_ptr<RankManager> rank_manager,
    std::function<std::shared_ptr<Simulator> (void)> get_simulator,
    const std::vector<NodeID> &nodes_to_schedule
) {
    double scheduler_construct_time, scheduling_time;
    Timestamp s, e;
    s = RealTimeNow();
    DynamicScheduler scheduler(_task_predictor, _transmit_predictor,
                               example_GCG, rank_manager,
                               get_simulator);
    e = RealTimeNow();
    scheduler_construct_time = (double)(e - s) / 1000000;
    
    s = RealTimeNow();
    std::unordered_map<NodeID, Rank> scheduling_results = scheduler.RunDynamicScheduling(nodes_to_schedule);
    e = RealTimeNow();
    scheduling_time = (double)(e - s) / 1000000;

#ifdef TEST_SCHEDULING_ALGORITHM
    std::ostringstream os;
    os << "At current we have nr_unscheduled[" << this->get_nr_unscheduled() << "] "
       << "nr_scheduled[" << this->get_nr_scheduled() << "] "
       << "We schedule "
       << "nr_nodes_to_be_scheduled[" << scheduler.nr_nodes_to_be_scheduled << "] "
       << "nr_ranks[" << scheduler.nr_ranks << "] "
       << "DynamicScheduler construct time [" << scheduler_construct_time << "] "
       << "scheduling time [" << scheduling_time << "] "
       << "fast heu time[" << scheduler.fast_heru_time << "] "
       << "nr_node_acc_pair__to_traval[" << scheduler.nr_node_acc_pair__to_traval << "] "
       << "time__to_traval_all_node_acc_pair[" << scheduler.time__to__traval_all_node_acc_pair << "] ";
    std::cout << os.str() << std::endl;
#endif
}


std::shared_ptr<GCG_Adding_OpManagement>
Get_FullGCG() {
    return std::make_shared<Full_GCG>();
}

std::shared_ptr<GCG_Adding_OpManagement>
Get_SimpleFullGCG() {
    return std::make_shared<Simple_Full_GCG>();
}

std::shared_ptr<GCG_Adding_OpManagement>
Get_GCG_FromJson(const nlohmann::json &j) {
    std::string type = j.at("type").get<std::string>();
    if (type == "Simple_Full_GCG") {
        auto gcg = std::make_shared<Simple_Full_GCG>(j);
        return gcg;
    } else if (type == "Full_GCG") {
        auto gcg = std::make_shared<Full_GCG>(j);
        return gcg;
    } else
        assert(0);
    return nullptr;
}

