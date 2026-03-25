#pragma once
#include <torch/csrc/jit/ir/ir.h>
#include <butil/memory/ref_counted.h>
#include "worker/worker_job_executor.h"
#include <acl/acl.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "common/event_pool.h"

namespace ors {

class Graph;

enum OpStatus {
    OP_NOT_READY = 0,
    OP_READY = 1,
    OP_RUNNING = 2,
    OP_COMPLETED = 3
};

// struct TensorMeta {
//     std::unordered_map<int, size_t> distribution; 
//     size_t total_bytes() const {
//         size_t sum = 0;
//         for (auto& kv : distribution) sum += kv.second;
//         return sum;
//     }
// };

class Operator:
    public WorkerJob, 
    public butil::RefCountedThreadSafe<Operator> {
public:
    friend class Graph;
    friend class HEFTScheduler;

    Operator(std::string op_name, 
             torch::jit::Node* node,
             uint32_t in_degree,
             std::string model_id, 
             uint64_t op_id,
             std::vector<int64_t> unresolved_tensor_indices = {},
             std::set<int64_t> dependency_indices = {},
             Graph* own_graph = nullptr);
    ~Operator() {}
    void reset() {
        // if(strcmp(_node->kind().toQualString(),
        //            "prim::GCG_get_native_device") == 0) {
        //     // _status = OpStatus::OP_READY;
        //     return;
        // }
        // if (_node->kind() == torch::jit::prim::Constant) {
        //     return;
        // }
        // _input_map.clear();
        // _output_map.clear();
        // // _input_meta.clear();
        // _has_transfer_event = false;
        // _status = OpStatus::OP_NOT_READY;
        // _in_degree.store(_unresolved_tensor_indices.size(), std::memory_order_release);
        if (_following_event != nullptr) {
            g_event_pool->push(_device.index(), _following_event);
            _following_event = nullptr;
        }
    }

    uint64_t id() {return _op_id;}

    OpStatus status();
    void set_status(OpStatus status);

    void set_own_graph(Graph* graph);

    void set_device(c10::Device device);
    c10::DeviceIndex device_idx() {
        return _device.index();
    }
    c10::DeviceType device_type() {
        return _device.type();
    }


    torch::jit::Node* get_jit_node();

    int32_t decrease_in_degree();
    int32_t get_in_degree();

    std::string get_op_name();
    uint64_t get_graph_id();
    std::string get_model_id() {return _model_id;}

    std::vector<int64_t> get_unresolved_tensor_indices() {
        std::unique_lock<std::mutex> lck(_mutex);
        return _unresolved_tensor_indices;
    }
    void add_child_op(Operator* child);
    void add_parent_op(Operator* parent);
    std::vector<Operator*> get_parent_ops();

    torch::jit::IValue move_to_device(torch::jit::IValue& v, c10::Device& device);
    bool check_if_inputs_need_migration();
    // void migrate_inputs_to_device();

    bool prepare_input(torch::jit::Stack& stack);
    bool extract_output(torch::jit::Stack& stack);

    butil::Status execute() override;
    butil::Status execute_constant();
    butil::Status execute_noop();
    butil::Status execute_op();
    void tupleConstruct(torch::jit::Stack& stack, size_t num_inputs);
    void namedTupleConstruct(torch::jit::Stack& stack, c10::TypePtr tuple_type, size_t num_inputs);
    butil::Status execute_tuple_construct();
    int64_t normalizeIndex(int64_t idx, int64_t list_size);
    void tupleIndex(torch::jit::Stack& stack);
    butil::Status execute_tuple_index();
    void listConstruct(torch::jit::Stack& stack, const c10::Type& list_type, size_t num_inputs);
    butil::Status execute_list_construct ();
    butil::Status execute_native_device_getter();

    void on_finished(butil::Status& st) override;

    void try_copy_inputs(std::unordered_map<int64_t, torch::jit::IValue>& input_map);
    void register_output(int64_t output_id, torch::jit::IValue output_value);
    

    // compute bytes to migration when schedule op to target device
    size_t compute_value_migration_cost(torch::jit::IValue& iv, c10::Device& target);
    size_t compute_op_migration_cost(c10::Device& target_device);

    bool need_synchronization();
    void update_op_status();

    // 可以在CPU上执行算子，并且不需要迁移输入数据
    bool is_cpu_op() {
        switch (_node->kind()) {
            case torch::jit::prim::Constant:
            case torch::jit::prim::Param:
            case torch::jit::prim::Return:
            case torch::jit::prim::ListConstruct:
            case torch::jit::prim::TupleConstruct:
            case torch::jit::prim::TupleIndex:
            // case torch::jit::aten::t:
            // // case torch::jit::aten::reshape:
            // case torch::jit::aten::sym_size:
                return true;
            default:
                return false;
        }
    }

    bool is_graph_infra_op() {
        switch(_node->kind()) {
            case torch::jit::prim::Constant:
            case torch::jit::prim::Param:
            case torch::jit::prim::Return:
                return true;
            default:
                return false;
        }
    }

    bool is_container_or_meta_op() {
        switch (_node->kind()) {
            case torch::jit::prim::ListConstruct:
            case torch::jit::prim::TupleConstruct:
            case torch::jit::prim::TupleIndex:
                return true;
            default:
                return false;
        }
    }

    bool is_event_recorded() {
        return _is_recorded;
    }
    bool has_transfer_event() {
        return _has_transfer_event;
    }

    // void record_compute_completion(at::cuda::CUDAStream& stream);
    void record_compute_completion();
    void record_transfer_completion(c10_npu::NPUStream& stream);

    // void wait_for_compute_ready(at::cuda::CUDAStream& stream) {
    //     _compute_ready.block(stream);
    // }

    void wait_for_op_ready(Operator* parent_op) {
        if (parent_op->is_event_recorded() &&
            parent_op->_assigned_stream != this->_assigned_stream) {
            // Make this op's stream wait until parent's compute event fires
            aclrtStreamWaitEvent(this->_assigned_stream.value().stream(),
                                 parent_op->_compute_ready);
        }
    }

    // void wait_for_transfer_ready(at::cuda::CUDAStream& stream) {
    //     _transfer_ready.block(stream);
    // }

    aclrtEvent get_completion_event() {
        return _compute_ready;
    }

    aclrtEvent get_transfer_event() {
        return _transfer_ready;
    }

    Operator* clone() {
        auto n_node = new Operator(this->_op_name,
                                    this->_node, 
                                    this->_node->inputs().size(),
                                    this->_model_id, 
                                    this->_op_id,
                                    this->_unresolved_tensor_indices,
                                    this->_dependency_indices);
        n_node->_operator_index = this->_operator_index;
        n_node->_function = this->_function;
        n_node->_input_map = this->_input_map;
        n_node->_output_map = this->_output_map;
        n_node->_input_reg_index = this->_input_reg_index;
        n_node->_output_reg_index = this->_output_reg_index;
        n_node->_move_input_afer_use = this->_move_input_afer_use;
        n_node->_in_degree = this->_in_degree.load(std::memory_order_acquire);
        n_node->_status = this->_status;
        n_node->_input_value_ids = this->_input_value_ids;
        n_node->_input_uses = this->_input_uses;
        return n_node;
    }

    void synchronize(c10_npu::NPUStream& stream) {
        c10_npu::NPUGuard guard(stream.device_index());
        aclrtStreamWaitEvent(stream.stream(), _following_event);
    }

    bool synchronizable() {
        std::unique_lock<std::mutex> lck(_mutex);
        return _following_event != nullptr;
    }

    bool enable_synchronization() {
        std::unique_lock<std::mutex> lck(_mutex);
        _following_event = g_event_pool->pop(_device.index());
        if (!_following_event) return false;
        aclrtRecordEvent(_following_event, _assigned_stream.value().stream());
        return true;
    }

    std::optional<c10_npu::NPUStream> _assigned_stream = std::nullopt;
    aclrtEvent _compute_ready = nullptr;
private:
    OpStatus _status;
    std::string _op_name;
    torch::jit::Node* _node;
public:
    std::atomic<int32_t> _in_degree;
private:
    // std::atomic<int32_t> _in_degree;
    // operator identity
    std::string _model_id;
    uint64_t _op_id;
    // dependency index
    std::vector<int64_t> _unresolved_tensor_indices;
    std::set<int64_t> _dependency_indices;
    // owning graph
    Graph* _own_graph;
    // device that operator be scheduled
    c10::Device _device{c10::DeviceType::CPU};
    // input and output map
    std::unordered_map<int64_t, torch::jit::IValue> _input_map = {};
    // std::unordered_map<int64_t, TensorMeta> _input_meta;
    std::unordered_map<int64_t, torch::jit::IValue> _output_map = {};    
    // whether has data transfer
    bool _is_recorded = false;
    bool _has_transfer_event{false};
    aclrtEvent _transfer_ready = nullptr;

    // mutex
    std::mutex _mutex;
public:
    int _operator_index = -1;
    torch::jit::Operation _function;
    std::vector<int> _input_reg_index;
    std::vector<int> _output_reg_index;
    std::vector<bool> _move_input_afer_use;
    torch::jit::Stack _stack;
  

    // child operators
    std::vector<Operator*> _child_ops;
    std::vector<Operator*> _parent_ops;
    std::vector<Operator*> _ready_child_ops;

    std::vector<int> _input_value_ids;
    std::vector<int> _input_uses;

    // for compute synchronization
    aclrtEvent _following_event = nullptr;
};

}