#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <torch/torch.h>
#include <c10/core/DeviceGuard.h>
#include "master/operator.h"
#include "master/graph.h"
#include "master/rpool.h"
#include "common/timing.h"

#ifdef ENABLE_PROFILE_DATA
#include "master/latency_estimate.h"
#endif

#include "worker/stream_manager.h"

#include <torch/csrc/jit/ir/irparser.h>

#include <fstream>
#include <sstream>
#include <iostream>

namespace ors {

#ifdef ENABLE_PROFILE_DATA
extern LatencyEstimator g_latency_est;
#endif


extern std::unordered_set<int> g_inject_error_op_ids;
extern bool g_enable_op_reschedule;
std::mutex g_inject_error_mutex;
extern thread_local int dse_thread_idx;
extern thread_local Operator* dse_target;

std::shared_ptr<torch::jit::Code> cache_code = nullptr;

// extern void launch_gpu_print(const float* data, int num_elements, cudaStream_t stream, const char* msg);

Operator::Operator(std::string op_name,
                   torch::jit::Node* node,
                   uint32_t in_degree,
                   std::string model_id, 
                   uint64_t op_id,
                   std::vector<int64_t> unresolved_tensor_indices,
                   std::set<int64_t> dependency_indices,
                   Graph* own_graph):
    _op_name(op_name), 
    _node(node), 
    _in_degree(in_degree),
    _model_id(model_id),
    _op_id(op_id), 
    _unresolved_tensor_indices(unresolved_tensor_indices),
    _dependency_indices(dependency_indices),
    _own_graph(own_graph),
    _function(nullptr) {
    _status = OpStatus::OP_NOT_READY;
    _stack.reserve(std::max(_node->inputs().size(), _node->outputs().size()));
    _ready_child_ops.clear();
    _ready_child_ops.reserve(_node->outputs().size());
}

OpStatus Operator::status() {
    std::unique_lock<std::mutex> lck(_mutex);
    return _status;
}

void Operator::set_status(OpStatus status) {
    // std::unique_lock<std::mutex> lck(_mutex);
    _status = status;
}

// following function not thread-safe, lock before
void Operator::set_own_graph(Graph* graph) {
    _own_graph = graph;
}

void Operator::set_device(c10::Device device) {
    _device = device;
}

torch::jit::Node* Operator::get_jit_node() {
    return _node;
}

int32_t Operator::decrease_in_degree() {
    return _in_degree.fetch_sub(1, std::memory_order_release);
}

int32_t Operator::get_in_degree() {
    return _in_degree.load(std::memory_order_acquire);
}

std::string Operator::get_op_name() {
    return _op_name;
}

uint64_t Operator::get_graph_id() {
    if (_own_graph) {
        return _own_graph->gid();
    } else {
        CHECK(false);
        return 0;
    }
}

void Operator::add_child_op(Operator* child) {
    _child_ops.push_back(child);
}

void Operator::add_parent_op(Operator* parent) {
    _parent_ops.push_back(parent);
}

std::vector<Operator*> Operator::get_parent_ops() {
    return _parent_ops;
}

bool Operator::check_if_inputs_need_migration() {
    if (is_container_or_meta_op() || is_graph_infra_op()) return false;
    for (auto& input : _node->inputs()) {
        auto it = _input_map.find(input->unique());
        CHECK(it != _input_map.end());
        const auto& val = it->second;
        if (val.isTensor()) {
            auto t = val.toTensor();
            if (t.defined() && t.device() != _device) return true;
        } else if (val.isTensorList()) {
            for (const at::Tensor& t : val.toTensorList()) {
                if (t.defined() && t.device() != _device) return true;
            }
        }
    }
    return false;
}

// void Operator::migrate_inputs_to_device() {
//     int dst_dev_id = _device.index();
//     auto dst_rx_stream = g_stream_manager->get_rx_stream(dst_dev_id);
//     // c10::cuda::CUDAStreamGuard g_dst(dst_rx_stream);
//     // auto dst_tx_stream = g_stream_manager->get_tx_stream(dst_dev_id);
//     // c10::cuda::CUDAStreamGuard g_dst(dst_rx_stream);


//     for (auto& input : _node->inputs()) {
//         auto it = _input_map.find(input->unique());
//         CHECK(it != _input_map.end());
//         // 获取 IValue 的引用，直接修改它
//         c10::IValue& val = it->second;
//         if (val.isTensor()) {
//             at::Tensor t = val.toTensor();
//             // 1. 检查是否定义 (处理 Optional Tensor)
//             if (!t.defined()) continue;
//             // 2. 检查设备
//             if (t.device() != _device) {
//                 int src_dev_id = t.device().index();
//                 if (t.device().is_cuda()) { // 在GPU设备上
//                     // D2D (Device to Device) 场景
//                     auto src_tx_stream = g_stream_manager->get_tx_stream(src_dev_id);
                    
//                     // c10::cuda::CUDAStreamGuard g_src(src_tx_stream);
//                     at::cuda::setCurrentCUDAStream(src_tx_stream);
//                     at::cuda::setCurrentCUDAStream(dst_rx_stream);
//                     val = t.to(_device, true);
                    
//                 } else {  // 在CPU设备上
//                     CHECK(false);
//                     c10::cuda::CUDAStreamGuard g_dst(dst_rx_stream);
//                     val = t.to(_device);
//                 }
//             } 
//         } else if (val.isTensorList()) {
//             c10::List<at::Tensor> list = val.toTensorList();
//             bool list_need_migration = false;
//             for (const at::Tensor& t : list) {
//                 if (t.defined() && t.device() != _device) {
//                     list_need_migration = true;
//                     break;
//                 }
//             }

//             if (list_need_migration) {
//                 c10::List<at::Tensor> new_list;
//                 new_list.reserve(list.size()); // 预分配内存

//                 for (const at::Tensor& t : list) {
//                     if (t.defined() && t.device() != _device) {
//                         int src_dev_id = t.device().index();
//                         at::Tensor new_t;
//                         if (t.device().is_cuda()) {
//                             auto src_tx_stream = g_stream_manager->get_tx_stream(src_dev_id);
//                             at::cuda::setCurrentCUDAStream(src_tx_stream);
//                             at::cuda::setCurrentCUDAStream(dst_rx_stream);
//                             new_t = t.to(_device, true);
                            
//                         } else {
//                             CHECK(false);
//                             c10::cuda::CUDAStreamGuard g_dst(dst_rx_stream);
//                             new_t = t.to(_device);
//                         }
//                         new_list.push_back(new_t);
//                     } else {
//                         new_list.push_back(t);
//                     }
//                 }
//                 val = new_list;
//             }
//         }
//     }
//     record_transfer_completion(dst_rx_stream);
// }

bool Operator::prepare_input(torch::jit::Stack& stack) {
    size_t sz = _node->inputs().size();
    for (size_t i = 0; i < sz; ++i) {
        // 此处先不判断是否需要move，直接copy，问题是显存峰值会增加，后续可以加一个参数控制是否开启move输入的优化
        // 一张图执行完毕后要及时清理输入寄存器，避免占用过多显存
        /*
        
        bool moved = (_input_uses[i] == ++(_own_graph->_use_count[_input_value_ids[i]]));

        if (moved) {
            stack.emplace_back(std::move(_own_graph->reg(_input_reg_index[i])));
        } else {
            stack.emplace_back(_own_graph->reg(_input_reg_index[i]));
        }
        */
        // if (_op_name == "aten::linear") {
        //     std::cout << "prepare input for aten::linear, input reg index: " << _input_reg_index[i] << std::endl;
        // }
        stack.emplace_back(_own_graph->reg(_input_reg_index[i]));
        // std::atomic_thread_fence(std::memory_order_seq_cst);
        // int prev_use_count = _own_graph->_use_count[_input_value_ids[i]].fetch_add(1, std::memory_order_seq_cst);
        // // CHECK(prev_use_count < _input_uses[i]);
        // if (prev_use_count == (_input_uses[i] - 1)) {
        //     _own_graph->release_reg(_input_reg_index[i]); // clear input register after last use
        // }
    }
    return true;
}

bool Operator::extract_output(torch::jit::Stack& stack) {
    // CHECK(stack.size() >= _node->outputs().size());
    // for (size_t i = 0; i < _node->outputs().size(); ++i) {
    //     _output_map[_node->outputs()[i]->unique()] = stack[i];
    // }
    size_t sz = _node->outputs().size();
    for (size_t i = sz; i > 0; --i) {
        _own_graph->reg(_output_reg_index[i - 1]) = torch::jit::pop(stack);
    }
    return true;
}

void Operator::register_output(int64_t output_id, torch::jit::IValue output_value) {
    _output_map[output_id] = output_value;
}

butil::Status Operator::execute() {
    // LOG(INFO) << "begin to execute operator, op_name: " << _op_name;
    // if (_op_name == "aten::multinomial") {
    //     LOG(INFO) << "execute multinomial operator, op_id: " << _op_id;
    // }
    // if (_op_name == "aten::slice") {
    //     LOG(INFO) << "execute slice operator, op_id: " << _op_id;
    // }
    // nvtxRangePush(_op_name.c_str());
    butil::Status st;
    switch (_node->kind()) {
        case torch::jit::prim::Constant:
            st = execute_constant();
            break;
        case torch::jit::prim::Param:
        case torch::jit::prim::Return:
            st = execute_noop();
            break;
        case torch::jit::prim::ListConstruct:
            st = execute_list_construct();
            break;
        case torch::jit::prim::TupleConstruct:
            st = execute_tuple_construct();
            break;
        case torch::jit::prim::TupleIndex:
            st = execute_tuple_index();
            break;
        default:
            st = execute_op();
            break;
    }
    // if (_assigned_stream.has_value()) {
    //     record_compute_completion(_assigned_stream.value());
    // }
    
    // nvtxRangePop();
    return st;
    // on finished
    // on_finished(st);
    
}

butil::Status Operator::execute_constant() {
    _stack.push_back(torch::jit::toIValue(get_jit_node()->output()).value());
    extract_output(_stack);
    return butil::Status::OK();
}

butil::Status Operator::execute_noop() {
    return butil::Status::OK();
}

butil::Status Operator::execute_native_device_getter() {
    // torch::jit::Stack stack;
    // CHECK(prepare_input(stack));
    prepare_input(_stack);
    CHECK(_stack.size() == 0);
    torch::jit::push(
        _stack,
        c10::IValue(_device));
    extract_output(_stack);
    return butil::Status::OK();
}

butil::Status Operator::execute_op() {
    if(strcmp(_node->kind().toQualString(),
                   "prim::GCG_get_native_device") == 0) {
        return execute_native_device_getter();
    }
    
    // // prepare input
    // torch::jit::Stack stack;
    // CHECK(prepare_input(stack));
    prepare_input(_stack);
    // find operator impletation

    // if (_op_name == "aten::slice") {
    //     auto t = _stack[0].toTensor();
    //     std::cout << "tensor dim: " << t.dim() << std::endl;
    // }
    
    try {
        _function(_stack);
        // _own_graph->_own_model->_operator_table[_operator_index](stack);
    } catch (const std::exception& e) {
        // CHECK(false);
        // std::cout << "111";
        LOG(ERROR) << "Operator not matched, error:\n" << e.what();
        CHECK(false);
    }
        // CHECK(executed == true);
    // auto stream = c10::cuda::getCurrentCUDAStream(_device.index());
    // CHECK(stream == g_stream_manager->get_compute_stream(_device.index()));
    // record_compute_completion(stream);
    // extract output
    extract_output(_stack);
    // auto curr_stream = c10::cuda::getCurrentCUDAStream(_device.index());
    // record_compute_completion(curr_stream);

    // // 具有多个子节点的算子需要记录完成事件，以防后续子节点在不同的流上执行
    // if (_child_ops.size() > 1 && 
    //     _assigned_stream.has_value() &&
    //     _device.index() >= 0) {
    //     // record_compute_completion();
    //     enable_synchronization();
    // }

    return butil::Status::OK();
}

void Operator::tupleConstruct(torch::jit::Stack& stack, size_t num_inputs) {
    if (num_inputs > stack.size()) {
        TORCH_CHECK(false, "Invalid number of inputs: ", num_inputs);
    }
    switch (num_inputs) {
        case 0:
        stack.emplace_back(c10::ivalue::Tuple::create());
        break;
        case 1:
        stack.back() = c10::ivalue::Tuple::create(std::move(stack.back()));
        break;
        case 2: {
        auto tuple = c10::ivalue::Tuple::create(
            std::move(stack[stack.size() - 2]),
            std::move(stack[stack.size() - 1]));
        stack.pop_back();
        stack.back() = std::move(tuple);
        break;
        }
        case 3: {
        auto tuple = c10::ivalue::Tuple::create(
            std::move(stack[stack.size() - 3]),
            std::move(stack[stack.size() - 2]),
            std::move(stack[stack.size() - 1]));
        stack.pop_back();
        stack.pop_back();
        stack.back() = std::move(tuple);
        break;
        }
        default: {
        std::vector<torch::jit::IValue> elems{
            std::make_move_iterator(stack.end() - num_inputs),
            std::make_move_iterator(stack.end())};
        torch::jit::drop(stack, num_inputs - 1);
        stack.back() = c10::ivalue::Tuple::create(std::move(elems));
        break;
        }
    }
}

void Operator::namedTupleConstruct(torch::jit::Stack& stack, c10::TypePtr tuple_type, size_t num_inputs) {
    std::vector<torch::jit::IValue> elems{
        std::make_move_iterator(stack.end() - num_inputs),
        std::make_move_iterator(stack.end())};
    torch::jit::drop(stack, num_inputs);
    torch::jit::push(
        stack,
        c10::ivalue::Tuple::createNamed(std::move(elems), std::move(tuple_type)));
}

butil::Status Operator::execute_tuple_construct() {
    // prepare input
    // torch::jit::Stack stack;
    // CHECK(prepare_input(stack));
    prepare_input(_stack);
    bool named =
        _node->output()->type()->expectRef<c10::TupleType>().name().has_value();
    const auto& type = _node->output()->type()->expect<torch::jit::TupleType>();
    auto num_input = _node->inputs().size();
    if (named) {
        namedTupleConstruct(_stack, type, num_input);
    } else {
        tupleConstruct(_stack, num_input);
    }
    
    // extract output
    extract_output(_stack);
    // auto curr_stream = c10::cuda::getCurrentCUDAStream(_device.index());
    // record_compute_completion(curr_stream);


    // // 具有多个子节点的算子需要记录完成事件，以防后续子节点在不同的流上执行
    // if (_child_ops.size() > 1 && 
    //     _assigned_stream.has_value() &&
    //     _device.index() >= 0) {
    //     // record_compute_completion();
    //     enable_synchronization();
    // }

    return butil::Status::OK();
}

int64_t Operator::normalizeIndex(int64_t idx, int64_t list_size) {
  if (idx < 0) {
    // Handle negative indexing
    idx = list_size + idx;
  }
  return idx;
}

void Operator::tupleIndex(torch::jit::Stack& stack) {
  int64_t index = torch::jit::pop(stack).toInt();
  auto tuple = torch::jit::pop(stack).toTuple();
  auto norm_index =
      normalizeIndex(index, static_cast<int64_t>(tuple->elements().size()));
  if (norm_index < 0 ||
      norm_index >= static_cast<int64_t>(tuple->elements().size())) {
    throw std::out_of_range("Tuple list index out of range");
  }
  stack.emplace_back(tuple->elements()[norm_index]);
}

butil::Status Operator::execute_tuple_index() {
    // prepare input
    // torch::jit::Stack stack;
    // CHECK(prepare_input(stack));
    prepare_input(_stack);
    // execute op
    tupleIndex(_stack);
    // extract output
    extract_output(_stack);
    // record event
    // auto curr_stream = c10::cuda::getCurrentCUDAStream(_device.index());
    // record_compute_completion(curr_stream);
    // execute success

    // // 具有多个子节点的算子需要记录完成事件，以防后续子节点在不同的流上执行
    // if (_child_ops.size() > 1 && 
    //     _assigned_stream.has_value() &&
    //     _device.index() >= 0) {
    //     // record_compute_completion();
    //     enable_synchronization();
    // }

    return butil::Status::OK();
}

void Operator::listConstruct(
    torch::jit::Stack& stack,
    const c10::Type& list_type,
    size_t num_inputs) {
    // Structuring the implementation this way allows NRVO to avoid
    // move-constructing vals on its way onto the stack. Moving a List
    // isn't free.
    auto makeList =
        [](torch::jit::Stack& stack, const c10::Type& list_type, size_t num_inputs) {
            c10::List<torch::jit::IValue> vals(list_type.containedType(0));
            vals.reserve(num_inputs);
            for (size_t i = stack.size() - num_inputs; i < stack.size(); ++i) {
            vals.push_back(std::move(stack[i]));
            }
            torch::jit::drop(stack, num_inputs);
            return vals;
        };
    stack.emplace_back(makeList(stack, list_type, num_inputs));
}

butil::Status Operator::execute_list_construct() {
    // prepare input
    // torch::jit::Stack stack;
    prepare_input(_stack);
    // execute op
    const auto& type = _node->output()->type()->expectRef<torch::jit::ListType>();
    auto num_input = _node->inputs().size();
    listConstruct(_stack, type, num_input);
    // extract output
    extract_output(_stack);
    // record event
    // auto curr_stream = c10::cuda::getCurrentCUDAStream(_device.index());
    // record_compute_completion(curr_stream);


    // // 具有多个子节点的算子需要记录完成事件，以防后续子节点在不同的流上执行
    // if (_child_ops.size() > 1 && 
    //     _assigned_stream.has_value() &&
    //     _device.index() >= 0) {
    //     // record_compute_completion();
    //     enable_synchronization();
    // }

    return butil::Status::OK();
}


bool Operator::need_synchronization() {
    if (!_assigned_stream.has_value() || _device.index() < 0) {
        return false;
    }
    // 如果算子有多个子节点，并且子节点可能在不同的流上执行，就需要同步事件
    if (_child_ops.size() > 1) {
        return true;
    }
    // 如果算子只有一个子节点，但是子节点有多个父节点，那么也需要同步事件
    if (_child_ops.size() == 1) {
        Operator* child = _child_ops[0];
        if (child->_parent_ops.size() > 1) {
            return true;
        }
    }
    return false;
}

void Operator::update_op_status() {
    // LOG(INFO) << "operator: " << _op_name << " has been excuted successfully";
    // set_status(OpStatus::OP_COMPLETED);
    // std::vector<scoped_refptr<Operator>> ready_ops;

    // // 具有多个子节点的算子需要记录完成事件，以防后续子节点在不同的流上执行
    // if (need_synchronization()) {
    //     // record_compute_completion();
    //     enable_synchronization();
    // }

    for (auto& op : _child_ops) {
        int32_t prev_degree = op->decrease_in_degree();
        // CHECK(prev_degree >= 1);
        if (prev_degree == 1) {
            // op->set_status(OpStatus::OP_READY);
            if (dse_target != nullptr) {
                _ready_child_ops.push_back(op);
                // ready_ops.push_back(op);
            } else {
                dse_target = op;
            } 
            // ready_ops.push_back(op);
        }
    }
    if (_ready_child_ops.size() > 0) {
        g_rpool->rpool_push_operator(_ready_child_ops, _ready_child_ops.size(), dse_thread_idx);
    }
}

void Operator::on_finished(butil::Status& st) {
    // CHECK(st.ok()) << "execute op " << _op_name 
    //                << " failed, st: " << st;
    // if (_op_name == "aten::div") {
    //     LOG(INFO) << "div operator finished, st: " << st;
    // }
    if (_op_name.compare("prim::Return") == 0) {
        _own_graph->_v_map.clear();
        // _input_map.clear();
        _output_map.clear();
        auto done = _own_graph->get_done();
        done->status().swap(st);
        return done->Run();
    }
    // // provide input for child op
    // for (auto& op : _child_ops) {
    //     op->try_copy_inputs(_output_map);
    // }
    // // remove input for current op
    // _output_map.clear();
    // _input_map.clear();
    // update op status
    update_op_status();
}

void Operator::try_copy_inputs(std::unordered_map<int64_t, torch::jit::IValue>& input_map) {
    std::unique_lock<std::mutex> lck(_mutex);
    for (auto& input : input_map) {
        int64_t unique_id = input.first;
        const torch::jit::IValue& val = input.second;

        if (_dependency_indices.find(unique_id) != _dependency_indices.end()) {
            _input_map[unique_id] = val;
        }
    }
}

// compute bytes to migration when schedule op to target device
size_t Operator::compute_value_migration_cost(torch::jit::IValue& iv, c10::Device& target) {
    size_t total_bytes = 0;
    if (iv.isTensor()) {
        auto t = iv.toTensor();
        if (t.device() != target) {
            total_bytes += t.numel() * t.element_size();
        }
    } else if (iv.isTuple()) {
        for (auto elem : iv.toTuple()->elements()) {
            total_bytes += compute_value_migration_cost(elem, target);
        }
    }
    return total_bytes;
}

size_t Operator::compute_op_migration_cost(c10::Device& target_device) {
    size_t total = 0;
    for (auto& input : _input_map) {
        total += compute_value_migration_cost(input.second, target_device);
    }
    return total;
}

// void Operator::record_compute_completion(at::cuda::CUDAStream& stream) {
//     // CHECK(stream == _assigned_stream.value());
//     _compute_ready.record(stream);
//     _is_recorded = true;;
// }

void Operator::record_compute_completion() {
    _is_recorded = true;
    aclrtRecordEvent(_following_event, _assigned_stream.value().stream());
}

void Operator::record_transfer_completion(c10_npu::NPUStream& stream) {
    aclrtRecordEvent(_transfer_ready, stream.stream());
    _has_transfer_event = true;
}


}