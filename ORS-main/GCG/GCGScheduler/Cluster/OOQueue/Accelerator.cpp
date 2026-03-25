#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <torch/jit.h>

#include "Cluster/OOQueue/OOQueue.h"

void
AcceleratorsAgent::AddAccelerator(Rank rank, std::shared_ptr<Accelerator> acc) {
  this->accs[rank] = acc;
  acc->SetAgent(this);
}

void
AcceleratorsAgent::StartUp() {
  for (auto acc_: this->accs) {
    auto acc = acc_.second;
    acc->StartUp();
  }
}

void
AcceleratorsAgent::LeaveTraceFromRemote(Rank rank,
        Trace trace,
        std::shared_ptr<TracePayload> trace_payload,
        std::optional<IssuingID> hint_issuing_id) {
    this->accs[rank]->LeaveTraceFromRemote(trace, trace_payload, hint_issuing_id);
}

void
AcceleratorsAgent::EnqueueActions(Rank rank,
        std::optional<IssuingID> issuing_id,
        std::vector<std::shared_ptr<AccActionSpec>> &op_specs) {
    this->accs[rank]->EnqueueActions(issuing_id, op_specs);
}

void
AcceleratorsAgent::PurgeTrace(Rank rank,
        Trace trace) {
    this->accs[rank]->PurgeTrace(trace);
}

void
AcceleratorsAgent::WithdrawActions(Rank rank,
        IssuingID issuing_id) {
    this->accs[rank]->WithdrawActions(issuing_id);
}


// 如果一个进程中的不同线程操纵了不同设备，每个线程需要从这里拿到其操纵的设备到底是什么
thread_local c10::Device torch_device("cpu");

void
set_torch_device__thread(c10::Device device) {
  torch_device = device;
}

torch::RegisterOperators native_device_getter(
        "prim::GCG_get_native_device",
        []() {return torch_device;});

void
empty_bottomhalf(std::shared_ptr<InfoToBottomHalf> info,
        std::optional<IssuingID> issuing_id,
        Accelerator *acc) {
}
