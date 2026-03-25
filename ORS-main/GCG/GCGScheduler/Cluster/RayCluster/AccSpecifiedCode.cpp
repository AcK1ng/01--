#include "Base.h"
#include "Cluster/OOQueue/OOQueue.h"



OOQueueActionRet
init_cuda_aten_runtime_for_raycluster(std::vector<std::shared_ptr<RegPayload>> &inputs,
        std::shared_ptr<TracePayload> trace_payload,
        std::shared_ptr<AccActionParamPayload> param_,
        std::optional<IssuingID> issuing_id,
        Accelerator *acc) {
    auto tensor = torch::tensor({1.0}, acc->GetTorchDevice()); // for CUDA init
    return {nullptr, nullptr, nullptr};
}


struct TransmitRecv_BottomHalfInfo: public InfoToBottomHalf {
    std::shared_ptr<struct TransmitInfo> transmit_info;
};
OOQueueActionRet
transmit_recv_for_raycluster(std::vector<std::shared_ptr<RegPayload>> &inputs,
        std::shared_ptr<TracePayload> trace_payload,
        std::shared_ptr<AccActionParamPayload> param_,
        std::optional<IssuingID> issuing_id,
        Accelerator* acc) {
    auto param = std::static_pointer_cast<struct TransmitInfo>(param_);
    auto alloced_recv_payload = std::static_pointer_cast<RegPayload>(trace_payload);

#ifndef NDEBUG
    std::cout << "I'm receiver rank[" << acc->GetRank() << "] "
              << "I get a TranmitID[" << param->transmit_id << "] "
              << "Sender[" << param->send_rank << "] "
              << "Receiver[" << param->recv_rank << "] "
              << std::endl;
#endif
    acc->SendTraceToRemoteAccelerator(param->send_rank, Trace({std::nullopt, param->transmit_id}), nullptr);

    param->param_to_master = std::make_shared<struct AccReportRecvDone>();
    param->param_to_master->transmit_id = param->transmit_id;
    acc->RecvPayload(param, alloced_recv_payload);
#ifndef NDEBUG
    std::cout << "I'm receiver rank[" << acc->GetRank() << "] "
              << "I receive done!"
              << std::endl;
#endif

    auto info_2_bottom = std::make_shared<struct TransmitRecv_BottomHalfInfo>();
    info_2_bottom->transmit_info = param;
    return {alloced_recv_payload, info_2_bottom, nullptr};
}
void
transmit_recv_for_raycluster__bottomhalf(std::shared_ptr<InfoToBottomHalf> info_,
        std::optional<IssuingID> issuing_id,
        Accelerator *acc) {
    auto info = std::static_pointer_cast<struct TransmitRecv_BottomHalfInfo>(info_);
    acc->SendEventToMaster(AccReportRecvDone, issuing_id, info->transmit_info->param_to_master);
}

OOQueueActionRet
transmit_send_for_raycluster(std::vector<std::shared_ptr<RegPayload>> &inputs,
        std::shared_ptr<TracePayload> trace_payload,
        std::shared_ptr<AccActionParamPayload> param_,
        std::optional<IssuingID> issuing_id,
        Accelerator* acc) {
    auto param = std::static_pointer_cast<struct TransmitInfo>(param_);

    acc->SendPayload(param, inputs[0]);

    return {nullptr, nullptr, nullptr};
}
