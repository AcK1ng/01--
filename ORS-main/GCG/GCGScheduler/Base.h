#ifndef BASE_H
#define BASE_H
#include <cstddef> // for size_t
#include <optional>
#include <tuple>
#include <vector>
#include <memory>
#include <cassert>
#include <chrono>
#include <unordered_map>
#include <mutex>
#include <utility>
#include <iostream>

#include <torch/torch.h>
#include <msgpack.hpp>
#include <json.hpp>



//#define STATIC_SCHEDULE
// #define UTILIZATION_DEBUG
// #define DEBUG_OOQUEUE
// #define DEBUG_RUNATENOP
// #define TEST_SIMULATOR
#define TEST_SCHEDULING_ALGORITHM


namespace nlohmann {
  // Allows serializing and deserializing contents behind a std::unique_ptr.
  // See also: https://github.com/nlohmann/json/issues/975
  template <typename T> struct adl_serializer<std::unique_ptr<T>> {
    template <typename BasicJsonType> static void
    to_json(BasicJsonType& json_value, const std::unique_ptr<T>& ptr) {
      if (ptr.get())
        json_value = *ptr;
      else
        json_value = nullptr;
    }
    template <typename BasicJsonType> static void
    from_json(const BasicJsonType& json_value, std::unique_ptr<T>& ptr) {
      if(!json_value.is_null()) {
        T inner_val = json_value.template get<T>();
        ptr = std::make_unique<T>(inner_val);
      }
    }
  };

  template <typename T> struct adl_serializer<std::shared_ptr<T>> {
    template <typename BasicJsonType> static void
    to_json(BasicJsonType& json_value, const std::shared_ptr<T>& ptr) {
      if (ptr.get())
        json_value = *ptr;
      else
        json_value = nullptr;
    }
    template <typename BasicJsonType> static void
    from_json(const BasicJsonType& json_value, std::shared_ptr<T>& ptr) {
      if(!json_value.is_null()) {
        T inner_val = json_value.template get<T>();
        ptr = std::make_shared<T>(inner_val);
      }
    }
  };

  template <typename T> struct adl_serializer<std::optional<T>> {
    template <typename BasicJsonType> static void
    to_json(BasicJsonType& json_value, const std::optional<T>& o) {
      if (o.has_value())
        json_value = o.value();
    }
    template <typename BasicJsonType> static void
    from_json(const BasicJsonType& json_value, std::optional<T>& o) {
      if (!json_value.is_null())
        o = json_value.template get<T>();
    }
  };
}


#define NR_STREAM 2

using PackageID = size_t; // Socket数量，比如双路机器，这个数字就是2
using CoreID = size_t; // 物理核心的ID，不是SMT的逻辑核心。这个空间包括双路机器，比如双路32核心共64核心，那这个数字取值是0-63
using LCoreID = size_t; // 物理核心内的逻辑核心ID，比如一个物理核心两个硬件线程，那LCoreID就是0或1

using NodeID = size_t; //这里的Node指计算图的Node, rather than the node in cluster
using Future = NodeID;
using TaskID = size_t;
using TaskNodeID = size_t;
using IssuingID = size_t;
using TransmitID = size_t;
using HostID = size_t;
using Rank = size_t;
using ComputeDomain = size_t;
using StreamID = size_t;
using Trace = std::tuple<std::optional<NodeID>,
                         std::optional<TransmitID>>;
namespace std {
    template <>
    struct hash<std::tuple<std::optional<size_t>, std::optional<size_t>>> {
        std::size_t operator()(const std::tuple<std::optional<size_t>, std::optional<size_t>> &p) const noexcept {
            auto &[v1, v2] = p;
            size_t t1 = v1.value_or(0), t2 = v2.value_or(0);
            std::size_t h1 = std::hash<size_t>{}(t1);
            std::size_t h2 = std::hash<size_t>{}(t2);
            return h1 ^ (h2 << 1); 
        }
    };
}
extern std::ostream&
operator<<(std::ostream &os, const Trace &Trace);

using DomainPair = std::tuple<ComputeDomain, ComputeDomain>; // 计算域之间的通信
using RankPair = std::tuple<Rank, Rank>;
namespace std {
    template <>
    struct hash<std::tuple<size_t, size_t>> {
        std::size_t operator()(const std::tuple<size_t, size_t> &p) const noexcept {
            std::size_t h1 = std::hash<size_t>{}(std::get<0>(p));
            std::size_t h2 = std::hash<size_t>{}(std::get<1>(p));
            return h1 ^ (h2 << 1);
        }
    };
}


namespace std {
    template <>
    struct hash<std::tuple<size_t, size_t, size_t>> {
        std::size_t operator()(const std::tuple<size_t, size_t, size_t> &p) const noexcept {
            std::size_t h1 = std::hash<size_t>{}(std::get<0>(p));
            std::size_t h2 = std::hash<size_t>{}(std::get<1>(p));
            std::size_t h3 = std::hash<size_t>{}(std::get<2>(p));
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

using NBytes = size_t;
using Timestamp = uint64_t;
using Duration = uint64_t;
using AccModel = std::string;


inline static Timestamp
RealTimeNow() {
  return std::chrono::time_point<std::chrono::steady_clock, std::chrono::nanoseconds>(std::chrono::steady_clock::now()).time_since_epoch().count();
}


template<typename T>
class my_priority_queue {
public:
    my_priority_queue(std::function<bool (T &l, T &r)> _comp): comp(_comp) { }
    std::vector<T> heap;
    std::function<bool (T &l, T &r)> comp;
    inline bool empty() { return this->heap.empty(); }
    inline T top() { return this->heap.front(); }
    inline void push(T &e) { this->heap.push_back(e); std::push_heap(this->heap.begin(), this->heap.end(), this->comp); }
    inline void pop() { std::pop_heap(this->heap.begin(), this->heap.end(), this->comp); this->heap.pop_back(); }
};


struct MyConstantPayload {
  MyConstantPayload() { }

  MyConstantPayload(const torch::jit::IValue &const_ival) {
    this->is_int = false;
    this->is_double = false;
    this->is_bool = false;
    this->is_tensor = false;
    if (const_ival.isInt()) {
        this->is_int = true;
        this->__i = const_ival.toInt();
    } else if (const_ival.isDouble()) {
        this->is_double = true;
        this->__d = const_ival.toDouble();
    } else if (const_ival.isBool()) {
        this->is_bool = true;
        this->__b = const_ival.toBool();
    } else if (const_ival.isTensor()) {
        this->is_tensor = true;
        this->serialized_data = std::make_shared<std::vector<char>>(torch::pickle_save(const_ival));
    } else {
        this->serialized_data = std::make_shared<std::vector<char>>(torch::pickle_save(const_ival));
    }
  }

  MyConstantPayload(std::vector<char> &serialized_data):
    MyConstantPayload(torch::pickle_load(serialized_data)) { }

  bool is_int;
  bool is_double;
  bool is_bool;
  bool is_tensor;

  int64_t __i;
  double __d;
  bool __b;
  std::shared_ptr<std::vector<char>> serialized_data;

  MSGPACK_DEFINE(
    is_int,
    is_double,
    is_bool,
    is_tensor,
    __i,
    __d,
    __b,
    serialized_data);

  friend void to_json(nlohmann::json& nlohmann_json_j, const MyConstantPayload& nlohmann_json_t) {
    nlohmann_json_j["is_int"] = nlohmann_json_t.is_int;
    nlohmann_json_j["is_double"] = nlohmann_json_t.is_double;
    nlohmann_json_j["is_bool"] = nlohmann_json_t.is_bool;
    nlohmann_json_j["is_tensor"] = nlohmann_json_t.is_tensor;
    if (nlohmann_json_t.is_int)
      nlohmann_json_j["__i"] = nlohmann_json_t.__i;
    else if (nlohmann_json_t.is_double)
      nlohmann_json_j["__d"] = nlohmann_json_t.__d;
    else if (nlohmann_json_t.is_bool)
      nlohmann_json_j["__b"] = nlohmann_json_t.__b;
    else if (nlohmann_json_t.serialized_data)
      nlohmann_json_j["serialized_data"] = nlohmann_json_t.serialized_data;
  }
  friend void from_json(const nlohmann::json& nlohmann_json_j, MyConstantPayload& nlohmann_json_t) {
    nlohmann_json_j.at("is_int").get_to(nlohmann_json_t.is_int);
    nlohmann_json_j.at("is_double").get_to(nlohmann_json_t.is_double);
    nlohmann_json_j.at("is_bool").get_to(nlohmann_json_t.is_bool);
    nlohmann_json_j.at("is_tensor").get_to(nlohmann_json_t.is_tensor);
    if (nlohmann_json_t.is_int)
      nlohmann_json_j.at("__i").get_to(nlohmann_json_t.__i);
    else if (nlohmann_json_t.is_double)
      nlohmann_json_j.at("__d").get_to(nlohmann_json_t.__d);
    else if (nlohmann_json_t.is_bool)
      nlohmann_json_j.at("__b").get_to(nlohmann_json_t.__b);
    else if (nlohmann_json_j.contains("serialized_data"))
      nlohmann_json_j.at("serialized_data").get_to(nlohmann_json_t.serialized_data);
  }
};



// 这个数据结构描述了类型，尤其是tuple包的tensor
// 初衷是为了让收发双方都知道传输大小
struct VariableDescriptor {
  VariableDescriptor() = default; 
  VariableDescriptor(const VariableDescriptor &old) {
    this->is_tuple = old.is_tuple;
    this->elems.reserve(old.elems.size());
    for (size_t i = 0; i < old.elems.size(); i++) {
      if (old.elems[i])
        this->elems.push_back(std::make_shared<VariableDescriptor>(*(old.elems[i])));
      else
        this->elems.push_back(nullptr);
    }
    this->is_tensor = old.is_tensor;
    this->shape = old.shape;
    this->numel = old.numel;
    this->dtype = old.dtype;
    this->serialized = old.serialized;
  }
  VariableDescriptor(std::vector<long int> shape, int dtype):
    is_tuple(0), is_tensor(1), shape(shape), dtype(dtype) {
      this->numel = 1;
      for (auto dim: this->shape)
        this->numel *= dim;
    }
  VariableDescriptor(const nlohmann::json &j) {
    if (j.contains("is_tuple")) {
      this->is_tuple = 1;
      this->is_tensor = 0;
      for (auto &e: j.at("elems")) {
        auto variable_descriptor = std::make_shared<struct VariableDescriptor>(e);
        this->elems.push_back(variable_descriptor);
      }
    } else if (j.contains("is_tensor")) {
      this->is_tuple = 0;
      this->is_tensor = 1;
      j.at("shape").get_to(this->shape);
      j.at("numel").get_to(this->numel);
      j.at("dtype").get_to(this->dtype);
    } else {
      this->is_tuple = 0;
      this->is_tensor = 0;
      j.at("serialized").get_to(this->serialized);
    }
  }
  int is_tuple;
  std::vector<std::shared_ptr<VariableDescriptor>> elems;

  int is_tensor;
  std::vector<long int> shape;
  size_t numel; // 标量元素个数
  int dtype;

  // is_tuple == false && is_tensor == false, 那么这就是个标量，具体数字在serialized里
  std::vector<char> serialized;

  MSGPACK_DEFINE(is_tuple, elems, is_tensor, shape, dtype, serialized);

  friend void to_json(nlohmann::json& j, const struct VariableDescriptor& p) {
    j = nlohmann::json();
    if (p.is_tuple) {
      j["is_tuple"] = 1;
      j["elems"] = {};
      for (size_t i = 0; i < p.elems.size(); i++) {
        if (p.elems[i]) {
          j["elems"][i] = nlohmann::json();
          to_json(j["elems"][i], *(p.elems[i]));
        } else
          j["elems"][i] = nullptr;
      }
    } else if (p.is_tensor) {
      j["is_tensor"] = 1;
      j["shape"] = p.shape;
      j["numel"] = p.numel;
      j["dtype"] = p.dtype;
    } else {
      j["serialized"] = p.serialized;
    }
  }

  friend void from_json(const nlohmann::json& j, struct VariableDescriptor& p) {
    p = VariableDescriptor(j);
  }
};

inline static void
for_each_leaf__of_two_tuple_tree(
  std::shared_ptr<VariableDescriptor> vd1, std::shared_ptr<VariableDescriptor> vd2,
  std::function<void (std::shared_ptr<VariableDescriptor> v1, std::shared_ptr<VariableDescriptor> v2)> visitor) {
  if (!vd1)
    return;
  if (vd1->is_tuple) {
    for (size_t i = 0; i < vd1->elems.size(); i++)
      for_each_leaf__of_two_tuple_tree(vd1->elems[i], vd2->elems[i], visitor);
  } else
    visitor(vd1, vd2);
}

inline static void
for_each_elem__in_tuple_vd(
  std::shared_ptr<VariableDescriptor> vd,
  std::function<void (std::shared_ptr<VariableDescriptor> &)> visitor) {
  if (!vd)
    return;
  visitor(vd);
  if (vd->is_tuple) {
    for (size_t i = 0; i < vd->elems.size(); i++)
      for_each_elem__in_tuple_vd(vd->elems[i], visitor);
  }
}

inline static NBytes
total_size_of_vd(std::shared_ptr<VariableDescriptor> vd) {
  NBytes ret = 0;
  auto predict_one = [&] (std::shared_ptr<VariableDescriptor> &_v) {
    if (!_v->is_tensor)
        return;
    ret += _v->numel * c10::elementSize((c10::ScalarType)_v->dtype);
  };
  for_each_elem__in_tuple_vd(vd, predict_one);
  return ret;
}

class AccActionBase {
public:
  AccActionBase() { }

  AccActionBase(int priority,
                size_t master_hint_sequence,
                size_t action_commit_id):
    priority(priority),
    master_hint_sequence(master_hint_sequence),
    action_commit_id(action_commit_id) { }

  int priority; // 越大越优先
  size_t master_hint_sequence; // 越小越优先
  size_t action_commit_id; // 越小越优先

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(AccActionBase,
    priority,
    master_hint_sequence,
    action_commit_id
  )

  MSGPACK_DEFINE(priority,
          master_hint_sequence,
          action_commit_id);
};

extern std::ostream&
operator<<(std::ostream &os, const struct AccActionBase &action);

static inline bool
operator<(const AccActionBase &action1, const AccActionBase &action2) {
  /* the big one will be executed first
     所以我们让优先级更低的更小
     如果operator<返回true，action1的优先级应该更低 */
  if (action1.priority != action2.priority)
    return action1.priority < action2.priority;
  if (action1.master_hint_sequence != action2.master_hint_sequence)
    return action1.master_hint_sequence > action2.master_hint_sequence;
  return action1.action_commit_id > action2.action_commit_id;
}

#define FOR_EACH_MASTER_EVENT__BY_WATCHDOG(_) \
  _(WatchdogSignAccIn) \
  _(WatchdogSignAccOut) \


#define FOR_EACH_MASTER_EVENT__BY_ACC(_) \
  _(AccReportNodeDone) \
  _(AccReportNodeFail) \
  _(AccRequestSend) \
  _(AccReportRecvDone) \
  _(AccReportCheckpointSettled) \


typedef enum {
#define DEF_MASTER_EVENT(e) e,
FOR_EACH_MASTER_EVENT__BY_WATCHDOG(DEF_MASTER_EVENT)
FOR_EACH_MASTER_EVENT__BY_ACC(DEF_MASTER_EVENT)
#undef DEF_MASTER_EVENT
} MasterEventEnum;
MSGPACK_ADD_ENUM(MasterEventEnum);

class MasterEventParamPayload { };
struct WatchdogSignAccIn: public MasterEventParamPayload {
  int compute_unit;
  int never_signout; // Reliable

  AccModel acc_model;
  std::string acc_name;
  HostID host_id;
  std::string resource;
  int local_device_id;
  Rank rank;
  ComputeDomain domain;
  size_t hbm_capability;
  
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(WatchdogSignAccIn,
          compute_unit,
          never_signout,
          acc_model,
          acc_name,
          host_id,
          resource,
          local_device_id,
          rank,
          domain,
          hbm_capability
        );

  MSGPACK_DEFINE(compute_unit,
          never_signout,
          acc_model,
          acc_name,
          host_id,
          resource,
          local_device_id,
          rank,
          domain,
          hbm_capability);
};

struct WatchdogSignAccOut: public MasterEventParamPayload {
  Rank rank;
  MSGPACK_DEFINE(rank);
};

struct AccReportNodeDone: public MasterEventParamPayload {
  NodeID node_id;
  TaskID task_id;
  TaskNodeID task_node_id;
  Duration running_time;
  Timestamp end;

  MSGPACK_DEFINE(node_id,
          task_id,
          task_node_id,
          running_time,
          end);
};

struct AccReportNodeFail: public MasterEventParamPayload {
  NodeID node_id;
  TaskID task_id;
  TaskNodeID task_node_id;

  MSGPACK_DEFINE(node_id,
          task_id,
          task_node_id);
};

struct AccRequestSend: public MasterEventParamPayload {
  TransmitID transmit_id;
  Rank recv_rank;
  std::shared_ptr<VariableDescriptor> variable_descriptor;

  MSGPACK_DEFINE(transmit_id, recv_rank, variable_descriptor);
};

struct AccReportRecvDone: public MasterEventParamPayload {
  TransmitID transmit_id;
  std::vector<std::pair<NBytes, Duration>> profiling;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(AccReportRecvDone,
    transmit_id,
    profiling)

  MSGPACK_DEFINE(transmit_id, profiling);
};

struct AccReportCheckpointSettled: public MasterEventParamPayload {
  NodeID node_id;

  MSGPACK_DEFINE(node_id);
};

#define FOR_EACH_ACC_ACTION(_) \
  _(HelloWorld) \
  _(InitATenRuntime) \
  _(DummyOutput) \
  _(RunATenOP) \
  _(UploadTensor) \
  _(Transmit_RequestSendToMaster) \
  _(Transmit_AllocTensor) \
  _(Transmit_Recv) \
  _(Transmit_Send) \
  _(SettledAsCheckpoint) \
  _(FetchCheckpoint) \


typedef enum {
#define DEF_ACC_ACTION(e) e,
FOR_EACH_ACC_ACTION(DEF_ACC_ACTION)
#undef DEF_ACC_ACTION
} AccActionEnum;
MSGPACK_ADD_ENUM(AccActionEnum);

NLOHMANN_JSON_SERIALIZE_ENUM(AccActionEnum, {
  {HelloWorld, "HelloWorld"},
  {InitATenRuntime, "InitATenRuntime"},
  {DummyOutput, "DummyOutput"},
  {RunATenOP, "RunATenOP"},
  {UploadTensor, "UploadTensor"},
  {Transmit_RequestSendToMaster, "Transmit_RequestSendToMaster"},
  {Transmit_AllocTensor, "Transmit_AllocTensor"},
  {Transmit_Recv, "Transmit_Recv"},
  {Transmit_Send, "Transmit_Send"},
  {SettledAsCheckpoint, "SettledAsCheckpoint"},
  {FetchCheckpoint, "FetchCheckpoint"},
})

class AccActionParamPayload { };
struct AccActionSpec {
  AccActionSpec() {}

  AccActionSpec(int priority,
                size_t master_hint_sequence,
                size_t action_commit_id,
                AccActionEnum op_enum,
                std::shared_ptr<AccActionParamPayload> op_param,
                std::vector<NodeID> input_ids,
                std::optional<NodeID> output_id,
                std::optional<Trace> trace_will_leave,
                StreamID stream_id,
                bool no_OOM,
                std::optional<Trace> wanted_trace,
                bool clear_trace):
    acc_action_base({priority, master_hint_sequence, action_commit_id}),
    op_enum(op_enum),
    op_param(op_param),
    input_ids(input_ids),
    output_id(output_id),
    trace_will_leave(trace_will_leave),
    stream_id(stream_id),
    no_OOM(no_OOM),
    wanted_trace(wanted_trace),
    clear_trace(clear_trace) {

  }
  AccActionBase acc_action_base;
  AccActionEnum op_enum;
  std::shared_ptr<AccActionParamPayload> op_param;
  std::vector<NodeID> input_ids;
  std::optional<NodeID> output_id;
  std::optional<Trace> trace_will_leave;
  StreamID stream_id;
  bool no_OOM;
  std::optional<Trace> wanted_trace;
  bool clear_trace;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(AccActionSpec,
    acc_action_base,
    op_enum,
    input_ids,
    output_id,
    trace_will_leave,
    stream_id,
    no_OOM,
    wanted_trace,
    clear_trace
  )

  MSGPACK_DEFINE(acc_action_base,
          op_enum,
          //op_param,
          input_ids,
          output_id,
          trace_will_leave,
          stream_id,
          no_OOM,
          wanted_trace,
          clear_trace);
};

extern std::ostream&
operator<<(std::ostream &os, const struct AccActionSpec &action_spec);

struct HelloWorld: public AccActionParamPayload {
  int i;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(HelloWorld,
          i);

  MSGPACK_DEFINE(i);
};

struct InitATenRuntime: public AccActionParamPayload {
  StreamID stream_id;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(InitATenRuntime, stream_id);
  MSGPACK_DEFINE();
};

struct DummyOutput: public AccActionParamPayload {
  int dummy_for_serialization;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(DummyOutput, dummy_for_serialization);
  MSGPACK_DEFINE();
};

struct RunATenOP: public AccActionParamPayload {
  std::string qualified_name;
  NodeID node_id;
  TaskID task_id;
  TaskNodeID task_node_id;
  std::string graph;

  size_t nr_op_inputs;
  std::vector<signed long int> op_input__to__action_input;
  std::vector<signed long int> op_input__to__consts;

  std::vector<struct MyConstantPayload> constants;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(RunATenOP,
          qualified_name,
          node_id,
          task_id,
          task_node_id,
          graph,
          nr_op_inputs,
          op_input__to__action_input,
          op_input__to__consts,
          constants);

  MSGPACK_DEFINE(qualified_name,
          node_id,
          task_id,
          task_node_id,
          graph,
          nr_op_inputs,
          op_input__to__action_input,
          op_input__to__consts,
          constants);
};

struct UploadTensor: public AccActionParamPayload {
  std::shared_ptr<std::vector<char>> f;
  NodeID node_id;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(UploadTensor,
          f,
          node_id);

  MSGPACK_DEFINE(f, node_id);
};

struct TransmitInfo: public AccActionParamPayload {
  TransmitID transmit_id;
  TransmitID transmit_id2;
  NodeID node__to_transmit;

  std::string send_resource;
  Rank send_rank;
  HostID send_host;
  ComputeDomain send_domain;
  
  std::string recv_resource;
  Rank recv_rank;
  HostID recv_host;
  ComputeDomain recv_domain;
  bool recv_for_settled_ckpt; // 指该transmit可能会影响到checkpoint的位置，用来做容错
  
  std::shared_ptr<struct AccReportRecvDone> param_to_master; // for receiver to record profiling data


  // for transmit manager in master
  bool has_requested_send, has_been_permitted, has_finished;
  Timestamp permit_time, finish_time;
  std::optional<IssuingID> issuing_id;
  // 这个信息是master里的，和node记录的shape不一样
  // node记录的shape是符号推理来的
  // 这里记的variable_descriptor是request send的时候返回来的，是运行时更准确的
  std::shared_ptr<VariableDescriptor> variable_descriptor;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(TransmitInfo,
          transmit_id,
          transmit_id2,
          node__to_transmit,
          send_resource,
          send_rank,
          send_host,
          send_domain,
          recv_resource,
          recv_rank,
          recv_host,
          recv_domain,
          recv_for_settled_ckpt,
          param_to_master,
          has_requested_send, has_been_permitted, has_finished,
          permit_time, finish_time,
          issuing_id,
          variable_descriptor
        );

  MSGPACK_DEFINE(transmit_id,
          transmit_id2,
          node__to_transmit,
          send_resource,
          send_rank,
          send_host,
          recv_resource,
          recv_rank,
          recv_host,
          recv_for_settled_ckpt);
};

extern std::ostream&
operator<<(std::ostream &os, const struct TransmitInfo &transmit_info);

struct Transmit_RequestSendToMaster: public TransmitInfo {
};

struct Transmit_AllocTensor: public TransmitInfo {
};

struct Transmit_Recv: public TransmitInfo {
};

struct Transmit_Send: public TransmitInfo {
};

struct TransmitSpec {
  TransmitID transmit_id;
  std::shared_ptr<VariableDescriptor> variable_descriptor;
  std::optional<IssuingID> hint_issuing_id;

  MSGPACK_DEFINE(transmit_id, variable_descriptor, hint_issuing_id);
};

struct SettledAsCheckpoint: public AccActionParamPayload {
  NodeID node_id;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(SettledAsCheckpoint,
          node_id);

  MSGPACK_DEFINE(node_id);
};

struct FetchCheckpoint: public AccActionParamPayload {
  int dummy_for_serialization;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(FetchCheckpoint, dummy_for_serialization);
  MSGPACK_DEFINE();
};

class Cluster;
class Simulator;
class Master {
public:
  virtual ~Master() { }
  virtual void DEBUG() {assert(0);}

  virtual void SetCluster(std::shared_ptr<Cluster> p) {
    this->cluster = p;
    this->cur_action_commit_id = 0;
  }
  virtual void Clear() {this->cluster = nullptr;}

  virtual void StartUp() {assert(0);}
  virtual void debugOutput() { }

  // Called by 'class Cluster' from Accelerators
  virtual void SendAccEvent(Rank from,
                            std::optional<IssuingID>,
                            Timestamp acc_timestamp,
                            MasterEventEnum,
                            std::shared_ptr<MasterEventParamPayload>) {assert(0);}

  // Called by 'class Cluster' from the Watchdog
  virtual void SendWatchdogEvent(MasterEventEnum, std::shared_ptr<MasterEventParamPayload>) {assert(0);}

  // Called directly by the Users
  virtual TaskID SubmitGraph(std::string root_graph,
          std::unordered_map<std::string, std::string> sub_graphs,
          std::unordered_map<int, std::string> symbol__to__symexpr,
          bool all_links_to_successor = false) {assert(0); return 0;}
  virtual void DropTask(TaskID) {assert(0);}
  virtual std::vector<Future> RunTask(TaskID task_id,
          std::vector<Future> inputs,
          std::optional<std::vector<Rank>> manual_assignment,
          std::vector<size_t> debug_output_i) {assert(0); return {};}
  virtual Future UploadTensor_ToCluster(std::vector<char> f, std::vector<long int> shape, int dtype) {assert(0); return 0;}
  virtual void DropFuture(Future) {assert(0);}
  virtual nlohmann::json ExportMasterStatus() {assert(0); return nullptr;}

  // Used for simulation
public:
  Master() { }
  Master(const Master &old) {
    this->cur_action_commit_id = old.cur_action_commit_id;
    this->fake_timestamp = this->Now();
    this->UseFakeTimestamp(true);
  }
  Master(const nlohmann::json &j): master_status_j(j) {
    j.at("cur_action_commit_id").get_to(this->cur_action_commit_id);
    j.at("now").get_to(this->fake_timestamp);
    this->UseFakeTimestamp(true);
  }
  virtual nlohmann::json
  ToJson() const {
    nlohmann::json j;
    j["cur_action_commit_id"] = this->cur_action_commit_id;
    j["now"] = this->Now();
    return j;
  }
  friend void to_json(nlohmann::json& nlohmann_json_j, const Master& nlohmann_json_t) {
    nlohmann_json_j = nlohmann_json_t.ToJson();
  }

  static std::shared_ptr<Master> GetMasterFromJson(const nlohmann::json &j);
  virtual std::shared_ptr<Simulator> GetFullClusterSimulator(std::shared_ptr<Master> self) {assert(0); return nullptr;}
  // 用于调度时并行模拟，从一个Master复制一堆一样的。这些clone的Master都共享一个RankManager，从Master的TaskManager为空、TaskPredictor为空
  virtual std::shared_ptr<Master> _clone() const {assert(0); return nullptr;}
protected:
  // 从主Master生成调度时模拟用的master，里面就包含了支持调度时模拟的所需的最小信息；哪怕信息可能不一致。
  // 主从Master共享一个RankManager，从Master的TaskManager为空、TaskPredictor为空
  virtual std::shared_ptr<Master> _dump_for_scheduling_time_simulation() const { return this->_clone(); }

  nlohmann::json master_status_j;

private:
  bool use_fake_timestamp;
  Timestamp fake_timestamp;

public:
  // 这两个函数是master指派算子、发射算子的逻辑。一个用途是master自己操纵自己调度；另一个用途是从外部模拟器中的master发射算子
  // 所以这两个函数首先实现成可以直接发射算子的样子，用于操纵模拟器；然后再和master自身的加锁情况适配起来。实现逻辑上的复用
  virtual void __inject_assigned_ranks__for_ops(NodeID node_id, Rank rank) {assert(0);}
  virtual void __issuing_ops(const std::vector<NodeID> &nodes_to_schedule,
                             const std::unordered_set<NodeID> &ops,
                             IssuingID issuing_id) {assert(0);}

public:
  void UseFakeTimestamp(bool yes) {
    this->use_fake_timestamp = yes;
  }
  void AdjustFakeTimestamp(Timestamp timestamp) {
    this->fake_timestamp = timestamp;
  }
  inline Timestamp Now() const {
    if (this->use_fake_timestamp)
      return this->fake_timestamp;
    return RealTimeNow();
  }

protected:
  // Called By the Master's logic
  inline void ActionStart() {
    this->action_mutex.lock();

#ifndef NDEBUG
    this->action_issuing = true;
#endif
  }

  void FreeCheckpoint(Rank rank, NodeID node_id);

  inline void IssueAction_ToCluster(
          Rank rank,
          std::optional<IssuingID> issuing_id_,
          AccActionEnum op_enum,
          std::shared_ptr<AccActionParamPayload> op_param,
          std::vector<NodeID> input_ids,
          std::optional<NodeID> output_id,
          std::optional<Trace> trace_will_leave,
          StreamID stream_id,
          int priority,
          bool no_OOM,
          std::optional<Trace> wanted_trace,
          bool clear_trace,
          size_t master_hint_sequence = 0) {
#ifndef NDEBUG
    assert(this->action_issuing);
#endif
    IssuingID issuing_id = issuing_id_.value_or(-1);
    auto issuing__to__actions_ = this->actions.find(rank);
    if (issuing__to__actions_ == this->actions.end()) {
      this->actions.insert({rank, {}});
      issuing__to__actions_ = this->actions.find(rank);
    }

    auto &issuing__to__actions = issuing__to__actions_->second;
    auto actions_ = issuing__to__actions.find(issuing_id);
    if (actions_ == issuing__to__actions.end()) {
      issuing__to__actions.insert({issuing_id, {}});
      actions_ = issuing__to__actions.find(issuing_id);
    }

    auto acc_action_spec = std::make_shared<AccActionSpec>();
    acc_action_spec->acc_action_base.priority = priority;
    acc_action_spec->acc_action_base.master_hint_sequence = master_hint_sequence;
    acc_action_spec->acc_action_base.action_commit_id = this->cur_action_commit_id;
    acc_action_spec->op_enum = op_enum;
    acc_action_spec->op_param = op_param;
    acc_action_spec->input_ids = input_ids;
    acc_action_spec->output_id = output_id;
    acc_action_spec->trace_will_leave = trace_will_leave;
    acc_action_spec->stream_id = stream_id;
    acc_action_spec->no_OOM = no_OOM;
    acc_action_spec->wanted_trace = wanted_trace;
    acc_action_spec->clear_trace = clear_trace;

    auto &actions = actions_->second;
    actions.push_back(acc_action_spec);
  }

  inline void PermitRecv_InCluster(
          Rank recv_rank,
          TransmitID transmit_id,
          std::shared_ptr<VariableDescriptor> variable_descriptor,
          std::optional<IssuingID> hint_issuing_id = std::nullopt) {
#ifndef NDEBUG
    assert(this->action_issuing);
#endif
    auto transmits_ = this->transmits_permitted.find(recv_rank);
    if (transmits_ == this->transmits_permitted.end()) {
      this->transmits_permitted.insert({recv_rank, {}});
      transmits_ = this->transmits_permitted.find(recv_rank);
    }

    auto transmit_spec = std::make_shared<TransmitSpec>();
    transmit_spec->transmit_id = transmit_id;
    transmit_spec->variable_descriptor = variable_descriptor;
    transmit_spec->hint_issuing_id = hint_issuing_id;

    auto &transmits = transmits_->second;
    transmits.push_back(transmit_spec);
  }

  inline void ActionCommit() {
#ifndef NDEBUG
    this->action_issuing = false;
#endif
    for (auto &i: this->actions) {
      auto rank = i.first;
      for (auto &j: i.second) {
        auto issuing_id_ = j.first;
        auto acc_action_specs = j.second;
        std::optional<IssuingID> issuing_id;
        if (issuing_id_ == -1)
          issuing_id = std::nullopt;
        else
          issuing_id = issuing_id_;
        this->IssueActions_ToCluster(rank, issuing_id, acc_action_specs);
      }
    }
    this->actions.clear();


    for (auto &i: this->transmits_permitted) {
      auto rank = i.first;
      auto transmits = i.second;
      this->PermitRecvs_InCluster(rank, transmits);
    }
    this->transmits_permitted.clear();

    this->cur_action_commit_id++;

    this->action_mutex.unlock();
  }

  void WithdrawActions_FromCluster(Rank, IssuingID);

private:
  void IssueActions_ToCluster(Rank,
                              std::optional<IssuingID>,
                              std::vector<std::shared_ptr<AccActionSpec>> &);
  void PermitRecvs_InCluster(Rank,
                             std::vector<std::shared_ptr<TransmitSpec>> &);

  std::shared_ptr<Cluster> cluster;

  size_t cur_action_commit_id;

#ifndef NDEBUG
  bool action_issuing;
#endif

  std::mutex action_mutex;
  std::unordered_map<
      Rank,
      std::unordered_map<
          IssuingID,
          std::vector<std::shared_ptr<AccActionSpec>>
      >
  > actions;
  std::unordered_map<
      Rank,
      std::vector<std::shared_ptr<TransmitSpec>>> transmits_permitted;
};

class Cluster {
public:
  virtual ~Cluster() { }
  virtual void SetMaster(std::shared_ptr<Master> p) {
    this->master = p;
  }
  virtual void Clear() {this->master = nullptr;}
  virtual void StartUp() {assert(0);}


  // Called by Master to handle the cluster
  virtual void IssueActions(Rank,
                            std::optional<IssuingID>,
                            std::vector<std::shared_ptr<AccActionSpec>> &) {assert(0);}
  virtual void PermitRecvs(Rank,
                           std::vector<std::shared_ptr<TransmitSpec>> &) {assert(0);}
  virtual void FreeCheckpoint(Rank, NodeID) {assert(0);}
  virtual void WithdrawActions(Rank, IssuingID) {assert(0);}


  // Called by native cluster or actor method to contact Master
  inline void SendAccEvent_ToMaster(Rank from,
                                    std::optional<IssuingID> issuing_id,
                                    Timestamp acc_timestamp,
                                    MasterEventEnum event,
                                    std::shared_ptr<MasterEventParamPayload> param) {
    this->master->SendAccEvent(from, issuing_id, acc_timestamp, event, param);
  }
  inline void SendWatchdogEvent_ToMaster(MasterEventEnum event,
                                         std::shared_ptr<MasterEventParamPayload> param) {
    this->master->SendWatchdogEvent(event, param);
  }

  virtual void debugOutput() { }
private:
  std::shared_ptr<Master> master;
};


class ClusterSys {
public:
  ClusterSys(std::shared_ptr<Cluster> cluster,
             std::shared_ptr<Master> master):
    cluster(cluster), master(master) {}
  virtual ~ClusterSys() {this->cluster->Clear(); this->master->Clear();}

  virtual void StartUp() {
    this->master->SetCluster(this->cluster);
    this->cluster->SetMaster(this->master);
    this->master->StartUp();
    this->cluster->StartUp();
  }
protected:
  std::shared_ptr<Cluster> cluster;
  std::shared_ptr<Master> master;
};



typedef enum {
  ExecuteAction,
  EmptyAction,
} StepResult;

class Simulator: public ClusterSys {
public:
  class ClusterStatus: public Cluster {
  public:
    virtual Timestamp NextActionEndTimestamp() = 0;
    virtual StepResult StepOneMoreActionEnd() = 0;
    virtual ComputeDomain GetMaxComputeDomain() = 0;
    virtual Rank GetMaxRank() = 0;
    virtual NBytes GetHBMUsage(ComputeDomain) = 0;
    virtual void check() = 0;

    void
    __register_action_done_callback(std::function<void (Rank, std::shared_ptr<const AccActionSpec>,
                                                        Timestamp end, Duration)> callback) {
      this->action_done_callback = callback;
    }

    void
    __register_task_predictor(std::function<Duration (AccModel, TaskID, TaskNodeID)> callback) {
      this->task_predictor = callback;
    }

    void
    __register_transmit_predictor(std::function<Duration (Rank, Rank, std::shared_ptr<VariableDescriptor>)> callback) {
      this->transmit_predictor = callback;
    }

    void
    __register_ask_node_shape(std::function<std::shared_ptr<VariableDescriptor> (NodeID)> callback) {
      this->ask_node_shape = callback;
    }

    void
    AdjustNow(Timestamp _now) {
      this->now = _now;
    }

    Timestamp now;
  protected:
    std::function<Duration (AccModel, TaskID, TaskNodeID)> task_predictor;
    std::function<Duration (Rank, Rank, std::shared_ptr<VariableDescriptor>)> transmit_predictor;
    std::function<std::shared_ptr<VariableDescriptor> (NodeID)> ask_node_shape;
    std::function<void (Rank, std::shared_ptr<const AccActionSpec>,
                        Timestamp end, Duration)> action_done_callback;
  };

  Simulator(std::shared_ptr<ClusterStatus> cluster_status,
            std::shared_ptr<Master> master):
    ClusterSys(cluster_status, master), fake_cluster(cluster_status) {
    this->master->SetCluster(this->cluster);
    this->cluster->SetMaster(this->master);
  }
  virtual ~Simulator() {
    this->fake_cluster = nullptr;
  }

  void
  check() {
    this->fake_cluster->check();
  }

  void
  AdvanceTime(Duration duration) {
    Timestamp start = this->master->Now();
    do {
        Timestamp t = this->fake_cluster->NextActionEndTimestamp();
        if (t == -1)
          break;

        if (start + duration < t)
          break;

        // t <= start + duration
        this->master->AdjustFakeTimestamp(t);
        this->fake_cluster->StepOneMoreActionEnd();
    } while(true);

    this->master->AdjustFakeTimestamp(start + duration);
    this->fake_cluster->AdjustNow(start + duration);
  }

  void InjectSchedulingPolicy(NodeID node_id, Rank rank) {
    this->master->__inject_assigned_ranks__for_ops(node_id, rank);
  }
  void CommitSchedulingPolicy(const std::vector<NodeID> &nodes_to_schedule,
                              const std::unordered_set<NodeID> &ops,
                              IssuingID issuing_id) {
    this->master->__issuing_ops(nodes_to_schedule, ops, issuing_id);
  }

  StepResult Step() {
    Timestamp t = this->fake_cluster->NextActionEndTimestamp();
    if (t == -1)
      return EmptyAction;
    this->master->AdjustFakeTimestamp(t);
    return this->fake_cluster->StepOneMoreActionEnd();
  }

  void debugOutput() {
    this->fake_cluster->debugOutput();
  }

  virtual void StartUp() final override {
    assert(0);
    // 这里不调用master和cluster的StartUP
    // 因为Master直接从外部导入，认为已经启动好了
    // ClusterStatus从Master派生而来，和Master的状态一致
    // this->master->StartUp();
    // this->cluster->StartUp();
  }

  Rank GetMaxRank() {
    return this->fake_cluster->GetMaxRank();
  }

  ComputeDomain GetMaxComputeDomain() {
    return this->fake_cluster->GetMaxComputeDomain();
  }

  NBytes GetHBMUsage(ComputeDomain domain) {
    return this->fake_cluster->GetHBMUsage(domain);
  }

  Timestamp Now() {
    return this->fake_cluster->now;
  }

  void
  __register_action_done_callback(std::function<void (Rank, std::shared_ptr<const AccActionSpec>,
                                                      Timestamp end, Duration)> callback) {
    this->fake_cluster->__register_action_done_callback(callback);
  }
protected:
  std::shared_ptr<ClusterStatus> fake_cluster;
};


#endif
