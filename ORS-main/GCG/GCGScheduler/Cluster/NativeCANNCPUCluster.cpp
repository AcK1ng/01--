#include "Base.h"
#include "Cluster/OOQueue/OOQueue.h"

#include <sstream>
#include <torch/torch.h>
#include <map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <unistd.h>
#include <dirent.h>

#include <hwloc.h>

#ifdef WITH_CANN
#include <Python.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

auto npu_synchronize = [] (long device_index) {
    c10_npu::NPUGuard device_guard(at::Device(at::DeviceType::PrivateUse1, device_index));
    c10_npu::npuSynchronizeDevice();
};
#endif

static size_t CPU_TP;
extern hwloc_topology_t topology;

inline static int
ceil_div(int a, int b) {
    // 向上整除
    return (a + b - 1) / b;
}

#ifdef WITH_CANN
#include "acl/acl.h"

#define npuErrchk(ans) { npuAssert((ans), __FILE__, __LINE__); }
inline void npuAssert(aclError code, const char *file, int line, bool abort=true) {
    if (code != ACL_SUCCESS) {
        fprintf(stderr,"GPUassert: %d\n", code);
        if (abort) exit(code);
    }
}

namespace {
    OOQueueActionRet init_aten_runtime(std::vector<std::shared_ptr<RegPayload>> &inputs,
                      std::shared_ptr<TracePayload> trace_payload,
                      std::shared_ptr<AccActionParamPayload> param_,
                      std::optional<IssuingID> issuing_id,
                      Accelerator *acc) {
        auto param = std::static_pointer_cast<struct InitATenRuntime>(param_);

        if (acc->GetResource() == "cpu") {
            torch::set_num_threads(CPU_TP);
        }

        auto tensor = torch::tensor({1.0}, acc->GetTorchDevice());

        return {nullptr, nullptr, nullptr};
    }
    
    void init_aten_runtime_bottomhalf(std::shared_ptr<InfoToBottomHalf> info,
                         std::optional<IssuingID> issuing_id,
                         Accelerator *acc) {
    }

    std::mutex payload_lock;
    std::map<TransmitID, std::shared_ptr<RegPayload>> payload__need_to_be_transmit;

    struct Transmit_RequestSendToMaster_BottomHalfInfo: public InfoToBottomHalf {
        TransmitID transmit_id;
        Rank recv_rank;
        std::shared_ptr<RegPayload> payload;
    };

    OOQueueActionRet transmit_request_send_to_master(std::vector<std::shared_ptr<RegPayload>> &inputs,
            std::shared_ptr<TracePayload> trace_payload,
            std::shared_ptr<AccActionParamPayload> param_,
            std::optional<IssuingID> issuing_id,
            Accelerator* acc) {
        auto param = std::static_pointer_cast<struct TransmitInfo>(param_);

        assert(param->send_domain != param->recv_domain);

        {
            std::lock_guard<std::mutex> holding_lock(payload_lock);
            payload__need_to_be_transmit[param->transmit_id] = inputs[0];
        }

        auto info_2_bottom = std::make_shared<struct Transmit_RequestSendToMaster_BottomHalfInfo>();
        info_2_bottom->transmit_id = param->transmit_id;
        info_2_bottom->recv_rank = param->recv_rank;
        info_2_bottom->payload = inputs[0];

        return {nullptr, info_2_bottom, nullptr};
    }
    
    void transmit_request_send_to_master__bottomhalf(std::shared_ptr<InfoToBottomHalf> info_,
            std::optional<IssuingID> issuing_id,
            Accelerator *acc) {
        auto info = std::static_pointer_cast<struct Transmit_RequestSendToMaster_BottomHalfInfo>(info_);
        auto param_to_master = std::make_shared<struct AccRequestSend>();
        param_to_master->transmit_id = info->transmit_id;
        param_to_master->recv_rank = info->recv_rank;
        param_to_master->variable_descriptor = IValue__to__VariableDescriptor(*(info->payload.get()));
        acc->SendEventToMaster(AccRequestSend, issuing_id, param_to_master);
    }

    struct TransmitRecv_BottomHalfInfo: public InfoToBottomHalf {
        std::shared_ptr<struct TransmitInfo> transmit_info;
    };

    OOQueueActionRet transmit_recv(std::vector<std::shared_ptr<RegPayload>> &inputs,
            std::shared_ptr<TracePayload> trace_payload,
            std::shared_ptr<AccActionParamPayload> param_,
            std::optional<IssuingID> issuing_id,
            Accelerator* acc) {
        auto param = std::static_pointer_cast<struct TransmitInfo>(param_);
        std::shared_ptr<RegPayload> ret;
        auto info_2_bottom = std::make_shared<struct TransmitRecv_BottomHalfInfo>();
        info_2_bottom->transmit_info = param;

        if (param->send_domain == param->recv_domain) {
            assert(param->send_resource == "cpu" && param->recv_resource == "cpu");
            ret = std::static_pointer_cast<RegPayload>(trace_payload);
        } else {
            acc->SendTraceToRemoteAccelerator(param->send_rank, Trace({std::nullopt, param->transmit_id}), nullptr, issuing_id);
            ret = std::static_pointer_cast<RegPayload>(trace_payload);

            std::shared_ptr<RegPayload> the_payload_from_sender;
            {
                std::lock_guard<std::mutex> holding_lock(payload_lock);
                the_payload_from_sender = payload__need_to_be_transmit.at(param->transmit_id);
                payload__need_to_be_transmit.erase(param->transmit_id);
            }

            param->param_to_master = std::make_shared<struct AccReportRecvDone>();
            param->param_to_master->transmit_id = param->transmit_id;
            auto &profiling_data = param->param_to_master->profiling;

            for_each_leaf__of_two_tuple_tree(*the_payload_from_sender, *ret,
                [&] (c10::IValue &src, c10::IValue &dst) {
                    if (src.isTensor()) {
                        Timestamp start = RealTimeNow();
                        dst.toTensor().copy_(src.toTensor());
                        npu_synchronize(acc->GetTorchDevice().index());
                        Timestamp end = RealTimeNow();
                        profiling_data.push_back({dst.toTensor().numel() * dst.toTensor().element_size(),
                                            end - start});
                    }
            });
        }

        return {ret, info_2_bottom, nullptr};
    }
    
    void transmit_recv__bottomhalf(std::shared_ptr<InfoToBottomHalf> info_,
            std::optional<IssuingID> issuing_id,
            Accelerator *acc) {
        auto info = std::static_pointer_cast<struct TransmitRecv_BottomHalfInfo>(info_);
        if (info->transmit_info->send_domain != info->transmit_info->recv_domain)
            acc->SendEventToMaster(AccReportRecvDone, issuing_id, info->transmit_info->param_to_master);
    }

    OOQueueActionRet transmit_send(std::vector<std::shared_ptr<RegPayload>> &inputs,
            std::shared_ptr<TracePayload> trace_payload,
            std::shared_ptr<AccActionParamPayload> param_,
            std::optional<IssuingID> issuing_id,
            Accelerator* acc) {
        auto param = std::static_pointer_cast<struct TransmitInfo>(param_);
        if (param->send_domain == param->recv_domain) {
            acc->SendTraceToRemoteAccelerator(param->recv_rank, Trace({std::nullopt, param->transmit_id}), inputs[0], issuing_id);
        } else {
            // do nothing
        }
        return {nullptr, nullptr, nullptr};
    }
}

#endif



class NativeCANNCPUAgent: public AcceleratorsAgent {
public:
    NativeCANNCPUAgent(Cluster *cluster) {
        this->affiliation_cluster = cluster;
    }

    void StartUp() override {
        for (auto acc_pair : this->accs) {
            auto acc = acc_pair.second;
#ifdef WITH_CANN
            acc->OverrideActionFun(InitATenRuntime,
                    init_aten_runtime, init_aten_runtime_bottomhalf);
            acc->OverrideActionFun(Transmit_RequestSendToMaster,
                    transmit_request_send_to_master, transmit_request_send_to_master__bottomhalf);
            acc->OverrideActionFun(Transmit_Recv,
                    transmit_recv, transmit_recv__bottomhalf);
            acc->OverrideActionFun(Transmit_Send, transmit_send);
#endif
        }
        this->AcceleratorsAgent::StartUp();
    }

    virtual void SendEventToMaster(Rank from,
            MasterEventEnum event,
            std::optional<IssuingID> issuing_id,
            std::shared_ptr<MasterEventParamPayload> param) override {
        affiliation_cluster->SendAccEvent_ToMaster(from, issuing_id, RealTimeNow(), event, param);
    }
    
    virtual void SendTraceToRemoteAccelerator(Rank rank,
            Trace trace,
            std::shared_ptr<TracePayload> trace_payload,
            std::optional<IssuingID> hint_issuing_id = std::nullopt) override {
        this->LeaveTraceFromRemote(rank, trace, trace_payload, hint_issuing_id);
    }
    
private:
    Cluster *affiliation_cluster;
};


class NativeCANNCPUCluster final: public Cluster {
public:
    virtual void StartUp() override {

#ifdef WITH_CANN
        std::cout << "\n====== Starting NativeCANNCPUCluster ======\n";

        hwloc_topology_init(&topology);
        hwloc_topology_set_type_filter (topology, HWLOC_OBJ_PCI_DEVICE, HWLOC_TYPE_FILTER_KEEP_ALL);
        hwloc_topology_set_type_filter (topology, HWLOC_OBJ_BRIDGE, HWLOC_TYPE_FILTER_KEEP_ALL);
        hwloc_topology_load(topology);

        std::unordered_map<PackageID, std::vector<CoreID>> rest_logical_cores;
        auto alloc_cores = [&](PackageID package_id, size_t need_cores) -> std::vector<CoreID> {
            std::vector<CoreID> ret;
            auto &cores = rest_logical_cores.at(package_id);
            if (cores.size() < need_cores)
                return ret;
            for (size_t i = 0; i < need_cores; i++) {
                ret.push_back(cores.back());
                cores.pop_back();
            }
            return ret;
        };

        std::string cpu_model;
        int nr_packages = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PACKAGE);
        int nr_cores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
        int nr_lp__per_core = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU) / nr_cores;
        for (PackageID package_id = 0; package_id < nr_packages; package_id++)
            rest_logical_cores[package_id] = {};

        for (PackageID package_id = 0; package_id < nr_packages; package_id++) {
            hwloc_obj_t package = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PACKAGE, package_id);
            for (int i = 0; i < package->infos_count; i++) {
                if (strcmp(package->infos[i].name, "CPUModel") == 0) {
                    cpu_model = package->infos[i].value;
                    break;
                }
            }

            CoreID core_id = 0;
            hwloc_obj_t core = NULL;
            while ((core = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_CORE, core)) != NULL) {
                if (hwloc_bitmap_intersects(package->cpuset, core->cpuset)) {
                    // 意思就是说，这个core_id属于这个package_id
                    rest_logical_cores.at(package_id).push_back(core_id);
                }
                core_id++;
            }
        }

        std::unordered_map<Rank, std::pair<PackageID, std::vector<CoreID>>> affined_package_id;

        if (nr_cores <= 32)
            CPU_TP = 4;
        else if (32 < nr_cores && nr_cores <= 96)
            CPU_TP = 8;
        else
            CPU_TP = 16;

        if (const char *cpu_tp_str = std::getenv("CPU_TP"))
            CPU_TP = std::stoi(cpu_tp_str);

        {
            std::ostringstream oss;
            oss << CPU_TP << "*(" << cpu_model << ")";
            cpu_model = oss.str();
        }
    
        this->accs_agent = std::make_shared<NativeCANNCPUAgent>(this);

        std::vector<std::pair<
            std::shared_ptr<Accelerator>,
            std::shared_ptr<struct WatchdogSignAccIn>
        >> accelerators;

        uint32_t nr_cudas = 0;

        Rank nr_ranks = 0;
        HostID nr_hosts = -1;
        ComputeDomain nr_domains = 0;
        std::string last_acc_model;
        {
            Py_Initialize();
            if (!Py_IsInitialized())
                exit(-1);
            // 初始化torch_npu
            int ret = PyRun_SimpleString(
                "import torch\n"
                "import torch_npu\n" 
            );
            if (ret != 0)
                exit(ret);

            npuErrchk(aclInit(nullptr));
            npuErrchk(aclrtGetDeviceCount(&nr_cudas));
            std::cout << "Detected " << nr_cudas << " CANN devices\n";
            
            for (int dev = 0; dev < nr_cudas; dev++) {
                npuErrchk(aclrtSetDevice(dev));
                const char *deviceName = aclrtGetSocName();

                size_t free, total_HBM;
                npuErrchk(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total_HBM));

                if (last_acc_model != deviceName) {
                    nr_hosts++;
                    last_acc_model = deviceName;
                }

                auto sign_in_param = std::make_shared<struct WatchdogSignAccIn>();

                sign_in_param->compute_unit = 1;
                sign_in_param->never_signout = 1; // for DEBUG
                sign_in_param->acc_model = last_acc_model;

                std::stringstream ss;
                ss << "npu:" << dev;
                sign_in_param->acc_name = ss.str();
                sign_in_param->host_id = nr_hosts;
                sign_in_param->resource = "npu";
                sign_in_param->local_device_id = dev;
                sign_in_param->rank = nr_ranks;
                sign_in_param->domain = nr_domains;
                sign_in_param->hbm_capability = total_HBM;
    
                hwloc_obj_t pcidev = hwloc_get_pcidev_by_busid(topology,
                                                            0,   //cuda_prop.pciDomainID,
                                                            0,   //cuda_prop.pciBusID,
                                                            0,   //cuda_prop.pciDeviceID,
                                                               0);
                hwloc_obj_t package = pcidev ? hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_PACKAGE, pcidev) : nullptr;
                PackageID package_id = -1;
                if (package)
                    package_id = package->os_index;
                assert(package_id != -1);

                // CUDA的队列，bottomhalf占一个硬件线程
                // 其他STREAM，每个STREAM一个硬件线程
                int need_lp = NR_STREAM + 1;

                std::vector<CoreID> affined_cores = alloc_cores(package_id, ceil_div(need_lp, nr_lp__per_core));
                assert(affined_cores.size() != 0);

                // 总共分配了这么多硬件线程
                int nr_lps = affined_cores.size() * nr_lp__per_core;

                // 最后一个硬件线程分给bottomhalf
                nr_lps--;
                std::pair<CoreID, LCoreID> bottomhalf_used_core = std::make_pair(affined_cores[nr_lps / nr_lp__per_core], nr_lps % nr_lp__per_core);

                std::vector<std::unordered_set<std::tuple<CoreID, LCoreID>>> top_half_used_cores(2);

                // 倒数第二个分给STREAM 1
                nr_lps--;
                top_half_used_cores[1] = {{affined_cores[nr_lps / nr_lp__per_core], nr_lps % nr_lp__per_core}};

                // 剩下的分给STREAM 0
                for (int i = 0; i < nr_lps; i++)
                    top_half_used_cores[0].insert(std::make_pair(affined_cores[i / nr_lp__per_core], i % nr_lp__per_core));

                // 这里写的这么扯淡主要是想照顾到SMT和非SMT，nr_lp__per_core不一样

                auto queue = GetOOQueue(NR_STREAM, top_half_used_cores, bottomhalf_used_core);

                auto acc = std::make_shared<Accelerator>(
                    std::move(queue),
                    nr_hosts,
                    "npu", 
                    last_acc_model,
                    dev,
                    nr_ranks,
                    total_HBM,
                    c10::kPrivateUse1,
                    dev
                );

                accelerators.push_back({acc, sign_in_param});

                affined_package_id.insert({sign_in_param->rank, {package_id, affined_cores}});

                this->accs_agent->AddAccelerator(nr_ranks, acc);

                nr_ranks++;
                nr_domains++;
            }

        }

        uint64_t total_memory = hwloc_get_root_obj(topology)->total_memory;
        total_memory = total_memory * 0.8;

        int nr_cpu_acc = 0;
        for (PackageID package_id = 0; package_id < nr_packages; package_id++) {
            do {
                // CPU的队列，bottomhalf占一个硬件线程
                // STREAM 0占CPU_TP个核心
                // STREAM 1占一个硬件线程
                int need_lp = nr_lp__per_core * CPU_TP + (NR_STREAM - 1) + 1;
                std::vector<CoreID> affined_cores = alloc_cores(package_id, ceil_div(need_lp, nr_lp__per_core));

                if (0 == affined_cores.size())
                    break;

                // 总共分配了这么多硬件线程
                int nr_lps = affined_cores.size() * nr_lp__per_core;

                // 最后一个硬件线程分给bottomhalf
                nr_lps--;
                std::pair<CoreID, LCoreID> bottomhalf_used_core = std::make_pair(affined_cores[nr_lps / nr_lp__per_core], nr_lps % nr_lp__per_core);

                std::vector<std::unordered_set<std::tuple<CoreID, LCoreID>>> top_half_used_cores(2);

                // 倒数第二个分给STREAM 1
                nr_lps--;
                top_half_used_cores[1] = {{affined_cores[nr_lps / nr_lp__per_core], nr_lps % nr_lp__per_core}};

                // 剩下的分给STREAM 0
                for (int i = 0; i < nr_lps; i++)
                    top_half_used_cores[0].insert(std::make_pair(affined_cores[i / nr_lp__per_core], i % nr_lp__per_core));

                auto queue = GetOOQueue(NR_STREAM, top_half_used_cores, bottomhalf_used_core);

                if (last_acc_model != cpu_model) {
                    nr_hosts++;
                    last_acc_model = cpu_model;
                }

                auto acc = std::make_shared<Accelerator>(
                    std::move(queue), 
                    nr_hosts,
                    "cpu", 
                    last_acc_model,
                    nr_cpu_acc,
                    nr_ranks,
                    total_memory,
                    c10::kCPU,
                    0
                );

                auto sign_in_param = std::make_shared<struct WatchdogSignAccIn>();

                sign_in_param->compute_unit = 1;
                sign_in_param->never_signout = 1; // for DEBUG
                sign_in_param->acc_model = acc->GetAccModel();

                std::stringstream ss;
                ss << "cpu:" << nr_cpu_acc;
                sign_in_param->acc_name = ss.str();
                sign_in_param->host_id = nr_hosts;
                sign_in_param->resource = "cpu";
                sign_in_param->local_device_id = nr_cpu_acc;
                sign_in_param->rank = nr_ranks;
                sign_in_param->domain = nr_domains;
                sign_in_param->hbm_capability = acc->GetHBMCapability();

                accelerators.push_back({acc, sign_in_param});

                affined_package_id.insert({sign_in_param->rank, {package_id, affined_cores}});

                this->accs_agent->AddAccelerator(nr_ranks, acc);

                nr_ranks++;
                nr_cpu_acc++;
            } while(1);
        }

        // 绑定一些core到队列线程上，那么剩下的线程用剩下的core
        std::multiset<CoreID> all_cores__need_to_be_reserverd;
        for (auto &[rank, _]: affined_package_id) {
            auto &[package_id, affined_cores] = _;
            for (CoreID core_id: affined_cores) {

                all_cores__need_to_be_reserverd.insert(core_id);
                std::cout << "reserve core " << core_id << " for rank " << rank << std::endl;
            }
        }
        hwloc_bitmap_t bitmap = hwloc_bitmap_alloc();
        hwloc_bitmap_t t = hwloc_bitmap_alloc();
        for (CoreID core_id = 0; core_id < nr_cores; core_id++) {
            if (all_cores__need_to_be_reserverd.contains(core_id))
                continue;
            hwloc_obj_t core = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, core_id);
            hwloc_bitmap_or(bitmap, core->cpuset, t);
            hwloc_bitmap_copy(t, bitmap);
        }
        hwloc_set_cpubind(topology, t, HWLOC_CPUBIND_PROCESS);
        hwloc_bitmap_free(bitmap);
        hwloc_bitmap_free(t);

        this->accs_agent->StartUp();   

        for (auto [acc, sign_in_param]:accelerators) {
            auto &[package_id, affined_cores] = affined_package_id.at(sign_in_param->rank);
            std::ostringstream oss;
            oss << "New device: "
                << "Host[" << sign_in_param->host_id << "] "
                << "Rank[" << sign_in_param->rank << "] "
                << "Domain[" << sign_in_param->domain << "] "
                << "Name[" << sign_in_param->acc_name << "] "
                << "HBM_Capability[" << sign_in_param->hbm_capability << "] "
                << "Type[" << sign_in_param->resource << "] "
                << "Model[" << sign_in_param->acc_model << "] "
                << "Affined_CPU_Package[" << package_id << "] ";
            oss << "Affined_Cores: ";
            for (CoreID core_id: affined_cores)
                oss << core_id << ",";
            std::cout << oss.str() << std::endl;

            this->SendWatchdogEvent_ToMaster(WatchdogSignAccIn, sign_in_param);
        }

        std::cout << "====== NativeCUDACPUCluster Initialization Complete ======\n";
#else
        exit(-1);
#endif
    }

    virtual void IssueActions(Rank rank,
            std::optional<IssuingID> issuing_id,
            std::vector<std::shared_ptr<AccActionSpec>>& op_specs) override {
        this->accs_agent->EnqueueActions(rank, issuing_id, op_specs);
    }

    virtual void PermitRecvs(Rank recv_rank,
            std::vector<std::shared_ptr<TransmitSpec>>& transmit_specs) override {
        for (auto transmit_spec : transmit_specs) {
            this->accs_agent->LeaveTraceFromRemote(recv_rank,
                    Trace({ std::nullopt, transmit_spec->transmit_id }),
                    transmit_spec->variable_descriptor,
                    transmit_spec->hint_issuing_id);
        }
    }

    virtual void FreeCheckpoint(Rank rank, NodeID node_id) {
        this->accs_agent->PurgeTrace(rank, Trace({node_id, std::nullopt}));
    }

private:
    std::shared_ptr<NativeCANNCPUAgent> accs_agent;  
};


std::shared_ptr<Cluster>
GetNativeCANNCPUCluster() {
    return std::make_shared<NativeCANNCPUCluster>();
}
