#ifndef OOQueue_H
#define OOQueue_H
#include "Base.h"

#include <optional>
#include <tuple>
#include <functional>
#include <vector>
#include <memory>
#include <cassert>
#include <map>
#include <unordered_map>

#include <torch/csrc/api/include/torch/types.h>
#include <c10/core/Device.h>



class AcceleratorsAgent; //forward declaration
class Accelerator; //forward declaration

// --------------for OOQueue Internal--------------------------
using RegID = NodeID;
using RegPayload = torch::jit::IValue; // RegPayload can not be nullopt
using TracePayload = void;
class InfoToBottomHalf { };

using OOQueueActionRet = std::tuple<std::shared_ptr<RegPayload>,
                                    std::shared_ptr<InfoToBottomHalf>,
                                    std::shared_ptr<TracePayload>>;

using OOQueueTopHalfActionFuncType =
std::function<OOQueueActionRet (std::vector<std::shared_ptr<RegPayload>> &,
                            std::shared_ptr<TracePayload>,
                            std::shared_ptr<AccActionParamPayload>,
                            std::optional<IssuingID>,
                            Accelerator *)>;

using OOQueueBottomHalfActionFuncType =
std::function<void (std::shared_ptr<InfoToBottomHalf>,
                    std::optional<IssuingID>,
                    Accelerator *)>;

extern
std::map<AccActionEnum, std::pair<OOQueueTopHalfActionFuncType,
                              OOQueueBottomHalfActionFuncType>> ActionEnum_to_ActionFunc;

// -------------------------------------------------------------

#include <iostream>
class OOQueue {
public:
    OOQueue(): op_enum__to__op_fun(ActionEnum_to_ActionFunc) { }

    void OverrideActionFun(AccActionEnum op_enum,
                       OOQueueTopHalfActionFuncType tophalf_op,
                       OOQueueBottomHalfActionFuncType bottomhalf_op) {
        this->op_enum__to__op_fun[op_enum] = {tophalf_op, bottomhalf_op};
    }
    inline std::pair<OOQueueTopHalfActionFuncType, OOQueueBottomHalfActionFuncType>
        GetActionFun(AccActionEnum op_enum) {
        return this->op_enum__to__op_fun[op_enum];
    }

    void SetAccelerator(Accelerator *acc) {this->acc = acc;}
    virtual void StartUp() {
        assert(this->acc != nullptr);
    }

    virtual void LeaveTraceFromRemote(Trace,
                                    std::shared_ptr<TracePayload>,
                                    std::optional<IssuingID> hint_issuing_id) {assert(0);}
    virtual void EnqueueActions(std::optional<IssuingID>, std::vector<std::shared_ptr<AccActionSpec>> &) {assert(0);}
    virtual void PurgeTrace(Trace) {assert(0);}
    virtual void WithdrawActions(IssuingID) {assert(0);}

protected:
    Accelerator *acc;

private:
    std::map<AccActionEnum, std::pair<OOQueueTopHalfActionFuncType,
                                  OOQueueBottomHalfActionFuncType>> op_enum__to__op_fun;
};

extern std::unique_ptr<OOQueue>
GetSimpleOOQueue(StreamID nr_stream);

extern std::unique_ptr<OOQueue>
GetOOQueue(StreamID nr_stream,
           std::vector<std::unordered_set<std::tuple<CoreID, LCoreID>>> top_half_used_cores,
           std::pair<CoreID, LCoreID> bottomhalf_used_core);

extern void
set_torch_device__thread(c10::Device);

class AcceleratorsAgent {
public:
    void AddAccelerator(Rank rank, std::shared_ptr<Accelerator> acc);

    // Need to be implemented for ActionFunc to contact other Accelerators or the Master
    virtual void StartUp();
    virtual void SendEventToMaster(Rank from,
                                   MasterEventEnum,
                                   std::optional<IssuingID>,
                                   std::shared_ptr<MasterEventParamPayload>) {assert(0);}
    virtual void SendTraceToRemoteAccelerator(Rank to,
                                              Trace,
                                              std::shared_ptr<TracePayload>,
                                              std::optional<IssuingID> hint_issuing_id = std::nullopt) {assert(0);}

    void LeaveTraceFromRemote(Rank rank,
            Trace trace,
            std::shared_ptr<TracePayload> trace_payload,
            std::optional<IssuingID> hint_issuing_id = std::nullopt);
    void EnqueueActions(Rank rank,
            std::optional<IssuingID> issuing_id,
            std::vector<std::shared_ptr<AccActionSpec>> &op_specs);
    void PurgeTrace(Rank, Trace);
    void WithdrawActions(Rank rank,
            IssuingID issuing_id);

    // For transmit
    virtual void SendPayload(std::shared_ptr<struct TransmitInfo> transmit_info,
            std::shared_ptr<RegPayload> payload) {
        assert(0);
    }
    virtual void RecvPayload(std::shared_ptr<struct TransmitInfo> transmit_info,
            std::shared_ptr<RegPayload> allocated_payload) {
        assert(0);
    }

    std::unordered_map<Rank, std::shared_ptr<Accelerator>> accs;
};


class Accelerator {
public:
    Accelerator(std::unique_ptr<OOQueue> queue,
                HostID host_id,
                const std::string resource,
                AccModel acc_model,
                int local_device_id,
                Rank rank,
                size_t hbm_capability,
                c10::DeviceType type,
                c10::DeviceIndex index = -1):
        queue(std::move(queue)),
        host_id(host_id),
        local_device_id(local_device_id),
        rank(rank),
        hbm_capability(hbm_capability),
        resource(resource),
        acc_model(acc_model),
        torch_device(c10::Device(type, index)) { }

    inline auto GetHostId() {return this->host_id;}
    inline auto GetLocalDeviceId() {return this->local_device_id;}
    inline auto GetResource() {return this->resource;}
    inline auto GetAccModel() {return this->acc_model;}
    inline auto GetRank() {return this->rank;}
    inline auto GetHBMCapability() {return this->hbm_capability;}
    inline auto GetTorchDevice() {return this->torch_device;}

    // Mainly used for agent to adopt remote action funs
    void SetAgent(AcceleratorsAgent *agent) {
        this->agent = agent;
    }
    void OverrideActionFun(AccActionEnum op_enum,
                       OOQueueTopHalfActionFuncType tophalf_op,
                       OOQueueBottomHalfActionFuncType bottomhalf_op) {
        this->queue->OverrideActionFun(op_enum, tophalf_op, bottomhalf_op);
    }
    void OverrideActionFun(AccActionEnum op_enum,
                       OOQueueTopHalfActionFuncType tophalf_op) {
        extern void
            empty_bottomhalf(std::shared_ptr<InfoToBottomHalf> info,
                    std::optional<IssuingID> issuing_id,
                    Accelerator *acc);
        this->OverrideActionFun(op_enum, tophalf_op, empty_bottomhalf);
    }

    void StartUp() {
        queue->SetAccelerator(this);
        queue->StartUp();
    }

    // Used for queue threads initialization
    void init_for_each_stream_thread() {
        set_torch_device__thread(this->GetTorchDevice());
    }

    inline void LeaveTraceFromRemote(Trace trace,
            std::shared_ptr<TracePayload> trace_payload,
            std::optional<IssuingID> hint_issuing_id = std::nullopt) {
        this->queue->LeaveTraceFromRemote(trace, trace_payload, hint_issuing_id);
    }
    inline void EnqueueActions(std::optional<IssuingID> issuing_id,
                    std::vector<std::shared_ptr<AccActionSpec>> &op_specs) {
        this->queue->EnqueueActions(issuing_id, op_specs);
    }
    inline void PurgeTrace(Trace trace) {
        this->queue->PurgeTrace(trace);
    }
    inline void WithdrawActions(IssuingID issuing_id) {
        this->queue->WithdrawActions(issuing_id);
    }

    void SendEventToMaster(MasterEventEnum e,
            std::optional<IssuingID> issuing_id,
            std::shared_ptr<MasterEventParamPayload> param) {
        this->agent->SendEventToMaster(this->GetRank(), e, issuing_id, param);
    }
    void SendTraceToRemoteAccelerator(Rank to,
            Trace trace,
            std::shared_ptr<TracePayload> trace_payload,
            std::optional<IssuingID> hint_issuing_id = std::nullopt) {
#ifdef DEBUG_OOQUEUE
        std::cout << "I'm rank[" << this->GetRank() << "] "
                  << "I want to leave a trace " << trace << " to rank[" << to << "] "
                  << std::endl;
#endif
        this->agent->SendTraceToRemoteAccelerator(to, trace, trace_payload, hint_issuing_id);
    }

    // For transmit
    void SendPayload(std::shared_ptr<struct TransmitInfo> transmit_info,
            std::shared_ptr<RegPayload> payload) {
        this->agent->SendPayload(transmit_info, payload);
    }
    void RecvPayload(std::shared_ptr<struct TransmitInfo> transmit_info,
            std::shared_ptr<RegPayload> allocated_payload) {
        this->agent->RecvPayload(transmit_info, allocated_payload);
    }


    std::unordered_map<std::tuple<TaskID, TaskNodeID>, torch::jit::Code> cached_code;

private:
    AcceleratorsAgent *agent;
    std::shared_ptr<OOQueue> queue;

    const HostID host_id;
    const std::string resource;
    const AccModel acc_model;
    const int local_device_id;
    const Rank rank;
    const size_t hbm_capability;
    c10::Device torch_device;
};

class Action {
public:
    AccActionSpec acc_action_spec;
    OOQueueTopHalfActionFuncType op_tophalf;
    OOQueueBottomHalfActionFuncType op_bottomhalf;
    std::vector<std::shared_ptr<RegPayload>> inputs; // When an input is nullptr, the input is un-ready.
    std::optional<IssuingID> issuing_id;
    int inner_priority;
    std::shared_ptr<TracePayload> trace_payload;
    
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Action,
        acc_action_spec,
        issuing_id,
        inner_priority
    )

    Action(std::optional<IssuingID> issuing_id,
       const AccActionSpec &opspec,
       OOQueueTopHalfActionFuncType op_tophalf,
       OOQueueBottomHalfActionFuncType op_bottomhalf):
        acc_action_spec(opspec),
        op_tophalf(op_tophalf),
        op_bottomhalf(op_bottomhalf),
        inputs(std::vector<std::shared_ptr<RegPayload>>(this->acc_action_spec.input_ids.size())),
        issuing_id(issuing_id),
        inner_priority(4),
        trace_payload(nullptr) {
        this->__update_ready();
    }

    bool provide_trace(Trace trace, std::shared_ptr<TracePayload> trace_payload) {
        if (this->acc_action_spec.wanted_trace.has_value()
                && this->acc_action_spec.wanted_trace.value() == trace) {
            this->acc_action_spec.wanted_trace = std::nullopt;
            this->trace_payload = trace_payload;
            this->__update_ready();
            if (this->acc_action_spec.clear_trace)
                return true; // the trace is cleared
            else
                return false;
        }
        return false;
    }

    void provide_input(RegID reg_id, std::shared_ptr<RegPayload> reg_payload) {
        int need_update = 0;
        for (size_t i = 0; i < this->acc_action_spec.input_ids.size(); i++)
            if (this->acc_action_spec.input_ids[i] == reg_id) {
                this->inputs[i] = reg_payload;
                need_update = 1;
            }
        if (need_update)
            this->__update_ready();
    }

    bool ready() { return this->__cached_ready; }
private:
    bool __cached_ready;
    void __update_ready() {
        this->__cached_ready = (this->acc_action_spec.wanted_trace == std::nullopt) && (this->__all_input_ready());
    }
    bool __all_input_ready() {
        return std::all_of(this->inputs.cbegin(),
                           this->inputs.cend(),
                           [](std::shared_ptr<RegPayload> input) { return input != nullptr; });
    }
};

inline static bool
operator<(const Action &op1, const Action &op2) {
    if (op1.inner_priority != op2.inner_priority)
        return op1.inner_priority < op2.inner_priority;
    return op1.acc_action_spec.acc_action_base < op2.acc_action_spec.acc_action_base;
}

inline static std::ostream&
operator<<(std::ostream &os, const Action &action) {
    nlohmann::json j = action;
    os << j.dump();
    return os;
}

inline static void
for_each_leaf__of_tuple_tree(
        c10::IValue &ivalue,
        std::function<void (c10::IValue &iv)> visitor) {
  if (ivalue.isTuple()) {
    auto elems = ivalue.toTupleRef().elements();
    for (auto &elem: elems)
      for_each_leaf__of_tuple_tree(elem, visitor);
  } else
    visitor(ivalue);
}

inline static void
for_each_leaf__of_two_tuple_tree(
        c10::IValue &ivalue1,
        c10::IValue &ivalue2,
        std::function<void (c10::IValue &iv1, c10::IValue &iv2)> visitor) {
  if (ivalue1.isTuple()) {
    auto elems_of_iv1 = ivalue1.toTupleRef().elements();
    auto elems_of_iv2 = ivalue2.toTupleRef().elements();
    for (size_t i = 0; i < elems_of_iv1.size(); i++)
      for_each_leaf__of_two_tuple_tree(elems_of_iv1[i], elems_of_iv2[i], visitor);
  } else
    visitor(ivalue1, ivalue2);
}

inline static void
for_each_elem__in_tuple_ivalue(
        c10::IValue &tuple,
        std::function<void (c10::IValue &)> visitor) {
  assert(tuple.isTuple());
  auto elems = tuple.toTupleRef().elements();
  for (auto &elem: elems)
    visitor(elem);
}

inline static std::shared_ptr<VariableDescriptor>
IValue__to__VariableDescriptor(c10::IValue &ivalue) {
  auto ret = std::make_shared<VariableDescriptor>();
  ret->is_tuple = 0;
  ret->is_tensor = 0;
  if (ivalue.isTuple()) {
    ret->is_tuple = 1;
    auto tuple_elem_visitor = [&](c10::IValue &elem) {
      ret->elems.push_back(IValue__to__VariableDescriptor(elem));
    };
    for_each_elem__in_tuple_ivalue(ivalue, tuple_elem_visitor);
  } else if (ivalue.isTensor()) {
    ret->is_tensor = 1;
    ret->shape = ivalue.toTensor().sizes().vec();
    ret->dtype = (int)ivalue.toTensor().dtype().toScalarType();
  } else {
    ret->serialized = torch::pickle_save(ivalue);
  }

  return ret;
}

inline static c10::IValue
VariableDescriptor__to__IValue(
        std::shared_ptr<VariableDescriptor> variable_descriptor,
        c10::Device device) {
  if (variable_descriptor->is_tuple) {
    std::vector<c10::IValue> elems;
    for (auto &elem_descriptor: variable_descriptor->elems)
      elems.push_back(VariableDescriptor__to__IValue(elem_descriptor, device));
    return c10::IValue(c10::ivalue::Tuple::create(elems));
  } else if (variable_descriptor->is_tensor) {
    auto options =
        c10::TensorOptions(device).dtype(c10::ScalarType(variable_descriptor->dtype));
    return torch::empty(variable_descriptor->shape, options);
  } else
    return torch::pickle_load(variable_descriptor->serialized);
}


#endif
