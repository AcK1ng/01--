#include "Cluster/OOQueue/OOQueue.h"

#include <iostream>

#include <torch/jit.h>
#include <torch/csrc/jit/ir/irparser.h>



std::map<AccActionEnum, std::pair<OOQueueTopHalfActionFuncType,
                              OOQueueBottomHalfActionFuncType>> ActionEnum_to_ActionFunc;

static void
OOQueueRegisterActionFunc(AccActionEnum op_enum,
                      OOQueueTopHalfActionFuncType tophalf_op,
                      OOQueueBottomHalfActionFuncType bottomhalf_op) {
    if (auto it = ActionEnum_to_ActionFunc.find(op_enum); it != ActionEnum_to_ActionFunc.end())
        assert(false);
    ActionEnum_to_ActionFunc[op_enum] = {tophalf_op, bottomhalf_op};
}

#define RegisterOP(op_enum, tophalf_op, bottomhalf_op) \
class op_enum ## _OOQueue_Register { \
public: \
    op_enum ## _OOQueue_Register() { \
        OOQueueRegisterActionFunc(op_enum, tophalf_op, bottomhalf_op); \
    } \
}; \
static op_enum ## _OOQueue_Register op_enum ## _OOQueue_Register_;


extern void
empty_bottomhalf(std::shared_ptr<InfoToBottomHalf> info,
        std::optional<IssuingID> issuing_id,
        Accelerator *acc);

static OOQueueActionRet
hello_world(std::vector<std::shared_ptr<RegPayload>> &inputs,
            std::shared_ptr<TracePayload> trace_payload,
            std::shared_ptr<AccActionParamPayload> param_,
            std::optional<IssuingID> issuing_id,
            Accelerator *acc) {
    auto param = std::static_pointer_cast<struct HelloWorld>(param_);

    std::ostringstream os;
    os << "HelloWorld "
       << "Rank[" << acc->GetRank() << "] "
       << "HostId[" << acc->GetHostId() << "] "
       << "LocalDeviceId[" << acc->GetLocalDeviceId() << "] "
       << "Resource[" << acc->GetResource() << "] "
       << "Param_from_master[" << param->i << "] "
       ;
    std::cout << os.str() << std::endl;
    return {nullptr, { }, nullptr};
}
static void
hello_world_bottomhalf(std::shared_ptr<InfoToBottomHalf> info,
                       std::optional<IssuingID> issuing_id,
                       Accelerator *acc) {
    std::cout << "Over!"
              << std::endl;
}
RegisterOP(HelloWorld, hello_world, hello_world_bottomhalf);


static OOQueueActionRet dummy_output(std::vector<std::shared_ptr<RegPayload>> &inputs,
        std::shared_ptr<TracePayload> trace_payload,
        std::shared_ptr<AccActionParamPayload> param_,
        std::optional<IssuingID> issuing_id,
        Accelerator* acc) {
    auto param = std::static_pointer_cast<struct DummyOutput>(param_);
    auto ret = std::make_shared<RegPayload>();

    return {ret, nullptr, nullptr};
}

RegisterOP(DummyOutput, dummy_output, empty_bottomhalf);


class MyInterpreter {
public:
    MyInterpreter(std::shared_ptr<torch::jit::Graph> graph,
                  torch::jit::Stack &stack) {
        this->graph = graph;
        
        std::cout << "----------------Input start--------------- " << std::endl;
        for (size_t input_idx = 0; input_idx < graph->inputs().size(); input_idx++) {
            auto input_value = graph->inputs()[input_idx];
            this->variables[input_value->unique()] = stack[input_idx];
            this->remaining_uses[input_value->unique()] = input_value->uses().size();
        }

#if 0
        for (auto &iv: stack) {
            nlohmann::json json = IValue__to__VariableDescriptor(iv);
            std::cout << json << std::endl;
        }
#endif

        stack.clear();
        std::cout << "----------------Input ready--------------- " << std::endl;
    }
    void print_ivalue(const torch::jit::IValue& ivalue, bool print_tensor_data = false) {
      if (ivalue.isTensor()) {
        at::Tensor tensor = ivalue.toTensor();
        if (print_tensor_data) {
          std::cout << tensor << std::endl;
        } else {
          std::cout << "Tensor[shape=" << tensor.sizes() 
                    << ", dtype=" << tensor.dtype() 
                    << ", device=" << tensor.device() << "]" << std::endl;
        }
      } else if (ivalue.isList()) {
        auto list = ivalue.toList();
        std::cout << "List[";
        for (size_t i = 0; i < list.size(); ++i) {
          if (i > 0) std::cout << ", ";
          print_ivalue(list.get(i), print_tensor_data);
        }
        std::cout << "]" << std::endl;
      } else if (ivalue.isTuple()) {
        auto tuple = ivalue.toTuple();
        if (tuple) {
          const auto& elements = tuple->elements();
          std::cout << "Tuple[";
          for (size_t i = 0; i < elements.size(); ++i) {
            if (i > 0) std::cout << ", ";
            print_ivalue(elements[i], print_tensor_data);
          }
          std::cout << "]" << std::endl;
        }
      } else if (ivalue.isDouble()) {
        std::cout << ivalue.toDouble() << std::endl;
      } else if (ivalue.isInt()) {
        std::cout << ivalue.toInt() << std::endl;
      } else if (ivalue.isBool()) {
        std::cout << (ivalue.toBool() ? "true" : "false") << std::endl;
      } else if (ivalue.isString()) {
        std::cout << ivalue.toStringRef() << std::endl;
      } else if (ivalue.isNone()) {
        std::cout << "None" << std::endl;
      } else {
        std::cout << "Unknown IValue type" << std::endl;
      }
    }

    void run(torch::jit::Stack &out_stack) {
        for (auto node: this->graph->nodes()) {
            std::cout << "----------------Node run start--------------- " << std::endl;

            torch::jit::Stack stack;
            for (auto input_value: node->inputs()) {
                torch::jit::push_one(stack, this->variables[input_value->unique()]);
            }


            std::cout << "Input: ";
            for (auto input_value: node->inputs()) {
                std::cout << input_value->unique() << ", ";
            }
            std::cout << std::endl;
            for (auto &iv: stack) {
                this->print_ivalue(iv);
            }

            std::cout << "Node: " << node->kind().toQualString() << std::endl;

            if (node->kind() == c10::prim::Constant) {
                auto ival = torch::jit::toIValue(node->output());
                torch::jit::push_one(stack, ival);
            } else if (node->kind() == c10::prim::TupleConstruct) {
                std::vector<torch::jit::IValue> elems;
                for (size_t i = 0; i < node->inputs().size(); i++) {
                    elems.push_back(torch::jit::pop(stack));
                }
                std::reverse(elems.begin(), elems.end());
                torch::jit::push_one(stack, torch::jit::IValue(c10::ivalue::Tuple::create(elems)));
            } else if (node->kind() == c10::prim::ListConstruct) {
                switch(node->output(0)->type()->expectRef<c10::ListType>().getElementType()->kind()) {
                    case c10::TypeKind::IntType: {
                        std::vector<int64_t> elems;
                        for (size_t i = 0; i < node->inputs().size(); i++) {
                            elems.push_back(torch::jit::pop(stack).toInt());
                        }
                        std::reverse(elems.begin(), elems.end());
                        torch::jit::push_one(stack, torch::jit::IValue(elems));
                        break;
                    }
                    case c10::TypeKind::TensorType: {
                        std::vector<torch::Tensor> elems;
                        for (size_t i = 0; i < node->inputs().size(); i++) {
                            elems.push_back(torch::jit::pop(stack).toTensor());
                        }
                        std::reverse(elems.begin(), elems.end());
                        torch::jit::push_one(stack, torch::jit::IValue(elems));
                        break;
                    }
                    default:
                        std::cout << "Unsupported list type" << std::endl;
                        exit(-1);
                }
            } else
                node->getOperation()(stack);

            std::cout << "Output: ";
            for (auto output_value: node->outputs()) {
                std::cout << output_value->unique() << ", ";
            }
            std::cout << std::endl;
            for (auto &iv: stack) {
                this->print_ivalue(iv);
            }

            for (auto node_input_value: node->inputs()) {
                this->remaining_uses[node_input_value->unique()]--;
                if (this->remaining_uses[node_input_value->unique()] == 0) {
                    this->variables.erase(node_input_value->unique());
                    this->remaining_uses.erase(node_input_value->unique());
                }
            }

            std::reverse(stack.begin(), stack.end());
            for (size_t output_idx = 0; output_idx < node->outputs().size(); output_idx++) {
                auto output_value = node->outputs()[output_idx];
                NodeID output_id = output_value->unique();
                this->variables[output_id] = torch::jit::pop(stack);
                this->remaining_uses[output_id] = output_value->uses().size();
            }

            std::cout << "----------------Node run end--------------- " << std::endl;
        }

        for (auto output_value: this->graph->outputs()) {
            out_stack.push_back(this->variables[output_value->unique()]);
        }
    }

private:
    std::shared_ptr<torch::jit::Graph> graph;
    std::unordered_map<NodeID, torch::jit::IValue> variables;
    std::unordered_map<NodeID, size_t> remaining_uses;
};

struct RunATenOP_BottomHalfInfo: public InfoToBottomHalf {
    NodeID node_id;
    TaskID task_id;
    TaskNodeID task_node_id;
    Duration running_time;
    Timestamp end;
};
static OOQueueActionRet
run_atenop(std::vector<std::shared_ptr<RegPayload>> &inputs,
           std::shared_ptr<TracePayload> trace_payload,
           std::shared_ptr<AccActionParamPayload> param_,
           std::optional<IssuingID> issuing_id,
           Accelerator* acc) {
    auto param = std::static_pointer_cast<struct RunATenOP>(param_);

    torch::DeviceGuard device_guard(acc->GetTorchDevice());
    torch::jit::Stack stack;
    stack.reserve(param->nr_op_inputs);

    for (size_t input_idx = 0; input_idx < param->nr_op_inputs; input_idx++) {
      if (param->op_input__to__action_input[input_idx] != -1) {
        torch::jit::push_one(stack, *(inputs[param->op_input__to__action_input[input_idx]]));
      } else /* if (param->op_input__to__consts[input_idx] != -1) */ {
        MyConstantPayload &const_payload = param->constants[param->op_input__to__consts[input_idx]];
        if (const_payload.is_int) {
            torch::jit::push_one(stack, const_payload.__i);
        } else if (const_payload.is_double) {
            torch::jit::push_one(stack, const_payload.__d);
        } else if (const_payload.is_bool) {
            torch::jit::push_one(stack, const_payload.__b);
        } else if (const_payload.is_tensor) {
            torch::jit::push_one(stack, torch::pickle_load(*(const_payload.serialized_data)));
        } else {
            torch::jit::push_one(stack, torch::pickle_load(*(const_payload.serialized_data)));
        }
      }
    }

    Timestamp start = RealTimeNow();
    if (param->qualified_name.compare("prim::GCG_Call_Submod") == 0) {


#ifdef DEBUG_RUNATENOP
        MyInterpreter my_interpreter(graph, stack);
        my_interpreter.run(stack);
#else

        if (!acc->cached_code.contains({param->task_id, param->task_node_id})) {
            auto graph = std::make_shared<torch::jit::Graph>();
            torch::jit::parseIR(param->graph, graph.get());
            acc->cached_code[{param->task_id, param->task_node_id}] = torch::jit::Code(graph, "<on-demand-func>");
        }
        torch::jit::InterpreterState(acc->cached_code.at({param->task_id, param->task_node_id})).run(stack);
#endif

    } else {
        const auto symbol = c10::Symbol::fromQualString(param->qualified_name);
        const auto ops = torch::jit::getAllSortedOperatorsFor(symbol);
        bool runned = 0;
        for (auto op: ops) {
            auto &formals = op->schema().arguments();
            if (formals.size() != stack.size())
                continue;
            try {
                op->getOperation()(stack);
                runned = 1;
            } catch(std::exception e) {
            }
            if (runned)
                break;
        }
        if (!runned)
            throw std::exception();
    }
    Timestamp end = RealTimeNow();

    std::shared_ptr<RegPayload> output;
    if (stack.size() == 0)
        output = nullptr;
    else if (stack.size() == 1) {
        output = std::make_shared<RegPayload>(std::move(stack[0]));
    } else
        throw std::exception();

    auto info_2_bottom = std::make_shared<struct RunATenOP_BottomHalfInfo>();
    info_2_bottom->node_id = param->node_id;
    info_2_bottom->task_id = param->task_id;
    info_2_bottom->task_node_id = param->task_node_id;
    info_2_bottom->running_time = end - start;
    info_2_bottom->end = end;

    return {output, info_2_bottom, nullptr};
}
static void
run_atenop_bottomhalf(std::shared_ptr<InfoToBottomHalf> info_,
                      std::optional<IssuingID> issuing_id,
                      Accelerator *acc) {
    auto info = std::static_pointer_cast<struct RunATenOP_BottomHalfInfo>(info_);
    auto param_to_master = std::make_shared<struct AccReportNodeDone>();
    param_to_master->node_id = info->node_id;
    param_to_master->task_id = info->task_id;
    param_to_master->task_node_id = info->task_node_id;
    param_to_master->running_time = info->running_time;
    param_to_master->end = info->end;
    // TODO: add acc Report Fail
    acc->SendEventToMaster(AccReportNodeDone, issuing_id, param_to_master);
}
RegisterOP(RunATenOP, run_atenop, run_atenop_bottomhalf);


static OOQueueActionRet
upload_tensor(std::vector<std::shared_ptr<RegPayload>> &inputs,
        std::shared_ptr<TracePayload> trace_payload,
        std::shared_ptr<AccActionParamPayload> param_,
        std::optional<IssuingID> issuing_id,
        Accelerator* acc) {
    auto param = std::static_pointer_cast<struct UploadTensor>(param_);

    Timestamp start = RealTimeNow();
    auto tensor = torch::pickle_load(*(param->f));
    tensor = tensor.toTensor().to(acc->GetTorchDevice());
    auto ret = std::make_shared<RegPayload>(std::move(tensor));
    Timestamp end = RealTimeNow();

    auto info_2_bottom = std::make_shared<struct RunATenOP_BottomHalfInfo>();
    info_2_bottom->node_id = param->node_id;
    info_2_bottom->running_time = end - start;
    info_2_bottom->end = end;
    return {ret, info_2_bottom, nullptr};
}
static void
upload_tensor__bottomhalf(std::shared_ptr<InfoToBottomHalf> info_,
        std::optional<IssuingID> issuing_id,
        Accelerator *acc) {
    auto info = std::static_pointer_cast<struct RunATenOP_BottomHalfInfo>(info_);
    auto param_to_master = std::make_shared<struct AccReportNodeDone>();
    param_to_master->node_id = info->node_id;
    param_to_master->running_time = info->running_time;
    param_to_master->end = info->end;
    acc->SendEventToMaster(AccReportNodeDone, issuing_id, param_to_master);
}
RegisterOP(UploadTensor, upload_tensor, upload_tensor__bottomhalf);


struct Transmit_RequestSendToMaster_BottomHalfInfo: public InfoToBottomHalf {
    TransmitID transmit_id;
    Rank recv_rank;
    std::shared_ptr<RegPayload> payload;
};
static OOQueueActionRet
transmit_request_send_to_master(std::vector<std::shared_ptr<RegPayload>> &inputs,
        std::shared_ptr<TracePayload> trace_payload,
        std::shared_ptr<AccActionParamPayload> param_,
        std::optional<IssuingID> issuing_id,
        Accelerator* acc) {
    auto param = std::static_pointer_cast<struct TransmitInfo>(param_);
    auto info_2_bottom = std::make_shared<struct Transmit_RequestSendToMaster_BottomHalfInfo>();
    info_2_bottom->transmit_id = param->transmit_id;
    info_2_bottom->recv_rank = param->recv_rank;
    info_2_bottom->payload = inputs[0];
    return {nullptr, info_2_bottom, nullptr};
}
static void
transmit_request_send_to_master__bottomhalf(std::shared_ptr<InfoToBottomHalf> info_,
        std::optional<IssuingID> issuing_id,
        Accelerator *acc) {
    auto info = std::static_pointer_cast<struct Transmit_RequestSendToMaster_BottomHalfInfo>(info_);
    auto param_to_master = std::make_shared<struct AccRequestSend>();
    param_to_master->transmit_id = info->transmit_id;
    param_to_master->recv_rank = info->recv_rank;
    param_to_master->variable_descriptor = IValue__to__VariableDescriptor(*(info->payload.get()));
    acc->SendEventToMaster(AccRequestSend, issuing_id, param_to_master);
}
RegisterOP(Transmit_RequestSendToMaster,
        transmit_request_send_to_master,
        transmit_request_send_to_master__bottomhalf);


static OOQueueActionRet
transmit_alloc_tensor(std::vector<std::shared_ptr<RegPayload>> &inputs,
        std::shared_ptr<TracePayload> trace_payload,
        std::shared_ptr<AccActionParamPayload> param_,
        std::optional<IssuingID> issuing_id,
        Accelerator* acc) {
    auto param = std::static_pointer_cast<struct TransmitInfo>(param_);

    auto variable_descriptor = std::static_pointer_cast<struct VariableDescriptor>(trace_payload);
    auto alloced_recv_payload = std::make_shared<RegPayload>(
            std::move(VariableDescriptor__to__IValue(variable_descriptor, acc->GetTorchDevice())));

    return {nullptr, nullptr, alloced_recv_payload};
}
RegisterOP(Transmit_AllocTensor, transmit_alloc_tensor, empty_bottomhalf);


struct TransmitRecv_BottomHalfInfo: public InfoToBottomHalf {
    TransmitID transmit_id;
};
static OOQueueActionRet
transmit_recv(std::vector<std::shared_ptr<RegPayload>> &inputs,
        std::shared_ptr<TracePayload> trace_payload,
        std::shared_ptr<AccActionParamPayload> param_,
        std::optional<IssuingID> issuing_id,
        Accelerator* acc) {
    auto param = std::static_pointer_cast<struct TransmitInfo>(param_);
    acc->SendTraceToRemoteAccelerator(param->send_rank, Trace({std::nullopt, param->transmit_id}), nullptr, issuing_id);
    auto alloced_recv_payload = std::static_pointer_cast<RegPayload>(trace_payload);

    assert(0);

    auto info_2_bottom = std::make_shared<struct TransmitRecv_BottomHalfInfo>();
    info_2_bottom->transmit_id = param->transmit_id;
    return {alloced_recv_payload, info_2_bottom, nullptr};
}
static void
transmit_recv__bottomhalf(std::shared_ptr<InfoToBottomHalf> info_,
        std::optional<IssuingID> issuing_id,
        Accelerator *acc) {
    auto info = std::static_pointer_cast<struct TransmitRecv_BottomHalfInfo>(info_);
    auto param_to_master = std::make_shared<struct AccReportRecvDone>();
    param_to_master->transmit_id = info->transmit_id;
    acc->SendEventToMaster(AccReportRecvDone, issuing_id, param_to_master);
}
RegisterOP(Transmit_Recv, transmit_recv, transmit_recv__bottomhalf);


static OOQueueActionRet
transmit_send(std::vector<std::shared_ptr<RegPayload>> &inputs,
        std::shared_ptr<TracePayload> trace_payload,
        std::shared_ptr<AccActionParamPayload> param_,
        std::optional<IssuingID> issuing_id,
        Accelerator* acc) {
    auto param = std::static_pointer_cast<struct TransmitInfo>(param_);

    assert(0);

    return {nullptr, nullptr, nullptr};
}
RegisterOP(Transmit_Send, transmit_send, empty_bottomhalf);


struct SettledAsCheckpoint_BottomHalfInfo: public InfoToBottomHalf {
    NodeID node_id;
};
static OOQueueActionRet
settled_as_checkpoint(std::vector<std::shared_ptr<RegPayload>> &inputs,
        std::shared_ptr<TracePayload> trace_payload,
        std::shared_ptr<AccActionParamPayload> param_,
        std::optional<IssuingID> issuing_id,
        Accelerator* acc) {
    auto param = std::static_pointer_cast<struct SettledAsCheckpoint>(param_);

    auto info_2_bottom = std::make_shared<struct SettledAsCheckpoint_BottomHalfInfo>();
    info_2_bottom->node_id = param->node_id;
    return {nullptr, info_2_bottom, inputs[0]};
}
static void
settled_as_checkpoint__bottomhalf(std::shared_ptr<InfoToBottomHalf> info_,
        std::optional<IssuingID> issuing_id,
        Accelerator *acc) {
    auto info = std::static_pointer_cast<struct SettledAsCheckpoint_BottomHalfInfo>(info_);
    auto param_to_master = std::make_shared<struct AccReportCheckpointSettled>();
    param_to_master->node_id = info->node_id;
    acc->SendEventToMaster(AccReportCheckpointSettled, issuing_id, param_to_master);
}
RegisterOP(SettledAsCheckpoint, settled_as_checkpoint, settled_as_checkpoint__bottomhalf);


static OOQueueActionRet
fetch_checkpoint(std::vector<std::shared_ptr<RegPayload>> &inputs,
        std::shared_ptr<TracePayload> trace_payload,
        std::shared_ptr<AccActionParamPayload> param_,
        std::optional<IssuingID> issuing_id,
        Accelerator* acc) {
    return {std::static_pointer_cast<RegPayload>(trace_payload), nullptr, nullptr};
}
RegisterOP(FetchCheckpoint, fetch_checkpoint, empty_bottomhalf);
