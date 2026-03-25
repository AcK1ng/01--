#include "Base.h"

#include "Master/FullMaster/FullMasterBase.h"
#include "Master/FullMaster/GCG.h"
#include "Master/FullMaster/TransmitManager.h"
#include "Master/FullMaster/RankManager.h"
#include "Master/FullMaster/ClusterStatus.h"


#include <shared_mutex>

class SimpleFullMaster final: public Master {
private:
    std::mutex issuing_id__lock;
    AliveIDs issuing_ids;

    std::shared_mutex rank_manager__lock;
    bool train_transmit_predictor;
    std::shared_ptr<RankManager> rank_manager;

    std::mutex GCG__lock;
    std::shared_ptr<GCG_Adding_OpManagement> GCG;

    std::mutex transmit_lock;
    std::shared_ptr<TransmitManager> transmit_manager;

    std::mutex scheduling__lock;

    bool __display_more;
    void DisplayMore(bool yes) { this->__display_more = yes; }

    bool schedule_more;
    void SetScheduleMore(bool yes) { this->schedule_more = yes; }

public:
    SimpleFullMaster() {
        this->rank_manager = std::make_shared<RankManager>();
        this->GCG = Get_FullGCG();
        this->transmit_manager = Get_SimpleTransmitManager();
        this->GCG->SetTrainTaskPredictor(true);
        this->SetTrainTransmitPredictor(true);
        this->UseFakeTimestamp(false);
        this->SetScheduleMore(true);
        this->DisplayMore(true);
        this->shared_init();
    }

    SimpleFullMaster(const SimpleFullMaster &another, bool share_rank_manager): Master(another) {
        this->issuing_ids = another.issuing_ids;
        if (share_rank_manager)
            this->rank_manager = another.rank_manager;
        else
            this->rank_manager = std::make_shared<RankManager>(*(another.rank_manager.get()));
        this->GCG = another.GCG->_clone();
        this->transmit_manager = another.transmit_manager->_clone();
        this->shared_init();
    }

    SimpleFullMaster(const nlohmann::json &j): Master(j) {
        j.at("issuing_ids").get_to(this->issuing_ids);
        j.at("rank_manager").get_to(this->rank_manager);
        j.at("train_transmit_predictor").get_to(this->train_transmit_predictor);
        this->GCG = Get_GCG_FromJson(j.at("GCG"));
        this->transmit_manager = Get_TransmitManager_FromJson(j.at("transmit_manager"));
        this->shared_init();
    }

    virtual nlohmann::json
    ToJson() const override {
        nlohmann::json j;
        j["type"] = "SimpleFullMaster";
        j["issuing_ids"] = this->issuing_ids;
        j["rank_manager"] = this->rank_manager;
        j["train_transmit_predictor"] = this->train_transmit_predictor;
        j["GCG"] = this->GCG;
        j["transmit_manager"] = this->transmit_manager;
        nlohmann::json parent = this->Master::ToJson();
        j.insert(parent.begin(), parent.end());
        return j;
    }

    virtual std::shared_ptr<Master>
    _clone() const override {
        auto ret = std::make_shared<SimpleFullMaster>(*this, true);
        ret->UseFakeTimestamp(true);
        ret->AdjustFakeTimestamp(this->Now());
        return ret;
    }
public:
    virtual void 
    DEBUG() override {
        Timestamp s, e;
        std::shared_ptr<Master> dumped_master_for_simulation;
        {
            std::lock_guard<std::mutex> tran_lock(transmit_lock);
            std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);
            s = RealTimeNow();
            dumped_master_for_simulation = this->_dump_for_scheduling_time_simulation();
            e = RealTimeNow();
            std::cout << "ScheduleTime master dumped "
                      << "spending " << ((double)(e - s) / 1000000) << "ms "
                      << std::endl;
        }

        std::cout << "===========DEBUG Scheduling Time Simulation Start===========" << std::endl;
        
        // 先拿调度时模拟器

        size_t nr_ops_simulated = 0;

        s = RealTimeNow();
        auto cloned_master = std::static_pointer_cast<SimpleFullMaster>(
            dumped_master_for_simulation->_clone());
        auto simulator = this->get_scheduletime_simulator(cloned_master);

        auto callback_for_action_done = [&, this](
            Rank r, std::shared_ptr<const AccActionSpec> a,
            Timestamp end, Duration duration) {
            nr_ops_simulated++;
            std::ostringstream oss;
            oss << std::setprecision(2)
                << "Timestamp[" << end << "] "
                << "Rank[" << r << "] "
                << "Action Done "
                << "Duration[" << duration << "] "
                << *a;
            std::cout << oss.str() << std::endl;
        };
        simulator->__register_action_done_callback(callback_for_action_done);
        e = RealTimeNow();
        std::cout << "ScheduleTime Simulator Construction Complete, "
                  << "spending " << ((double)(e - s) / 1000000) << "ms "
                  << std::endl;

        s = RealTimeNow();
        // 然后进行模拟
        while (simulator->Step() != EmptyAction)
            ;
        e = RealTimeNow();
        std::cout << "ScheduleTime Simulation Done, "
                  << nr_ops_simulated << " ops go beyound,  "
                  << "spending " << ((double)(e - s) / 1000000) << "ms "
                  << std::endl;

        simulator->debugOutput();

        std::cout << "===========DEBUG Scheduling Time Simulation End===========" << std::endl;
    }

    void
    SetTrainTransmitPredictor(bool yes) {
        this->train_transmit_predictor = yes;
    }

    virtual std::shared_ptr<Simulator>
    GetFullClusterSimulator(std::shared_ptr<Master> self) {
        std::unique_lock<std::shared_mutex> rank_lock(this->rank_manager__lock);
        std::lock_guard<std::mutex> tran_lock(transmit_lock);
        std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);
        std::lock_guard<std::mutex> issuing_id__lock(this->issuing_id__lock);
        std::lock_guard<std::mutex> __scheduling__lock(this->scheduling__lock);

        std::shared_ptr<RankManager> _rank_manager;
        this->master_status_j.at("rank_manager").get_to(_rank_manager);

        auto cluster_status = std::make_shared<ClusterStatusImpl>(_rank_manager);

        std::shared_ptr<TaskPerformancePredictor> task_predictor =
            GetTaskPerformancePredictorFromJson(this->master_status_j.at("GCG").at("task_predictor"));

        auto _task_predictor = [task_predictor](AccModel acc_model, TaskID task_id, TaskNodeID task_node_id) -> Duration {
            return task_predictor->predict(task_id, task_node_id, acc_model);
        };
        cluster_status->__register_task_predictor(_task_predictor);

        auto _transmit_predictor = [_rank_manager](Rank s, Rank r, std::shared_ptr<VariableDescriptor> vd) -> Duration {
            assert(vd);
            Duration ret = 0;
            auto predict_one = [&] (std::shared_ptr<VariableDescriptor> &_v) {
                if (!_v->is_tensor)
                    return;
                ret += _rank_manager->PredictTransmit(s, r,
                    _v->numel * c10::elementSize((c10::ScalarType)_v->dtype));
            };
            for_each_elem__in_tuple_vd(vd, predict_one);
            return ret;
        };
        cluster_status->__register_transmit_predictor(_transmit_predictor);

        auto _ask_node_shape = [this](NodeID node_id) -> std::shared_ptr<VariableDescriptor> {
            auto *node = this->GCG->__get_node(node_id);
            if (node->_.is_constant)
                return nullptr;
            return node->_.shape;
        };
        cluster_status->__register_ask_node_shape(_ask_node_shape);

        cluster_status->SetMaster(self);

        cluster_status->Init_Stage_2(this->Now(), *(this->transmit_manager), *(this->GCG));

        // 全集群模拟ClusterStatus中的Predictor与Rankmanager，和Master是独立的
        // 所以ClusterStatus中的假数据可以影响到Master
        this->GCG->SetTrainTaskPredictor(true);
        this->SetTrainTransmitPredictor(true);
        this->SetScheduleMore(true);
        this->DisplayMore(true);

        auto simulator = std::make_shared<Simulator>(cluster_status, self);
        cluster_status->Init_Stage_3(*(this->transmit_manager), *(this->GCG));
        return simulator;
    }

private:
    void
    shared_init() {
        auto timer = [this]() {
            return this->Now();
        };
        this->GCG->__register_timer(timer);
        this->transmit_manager->__register_timer(timer);

        auto transmit_issuing = [this](std::shared_ptr<struct TransmitInfo> transmit_info) {
            auto transmit_id1 = transmit_info->transmit_id;
            auto transmit_id2 = transmit_info->transmit_id2;
            auto node_id = transmit_info->node__to_transmit;
            auto send = transmit_info->send_rank;
            auto recv = transmit_info->recv_rank;
            auto issuing_id = transmit_info->issuing_id;

            if (transmit_info->send_domain != transmit_info->recv_domain) {
                this->IssueAction_ToCluster(send, issuing_id, Transmit_RequestSendToMaster, transmit_info,
                    {node_id}, std::nullopt, std::nullopt,
                    1, 0, true, std::nullopt, false, transmit_id1);
                this->IssueAction_ToCluster(send, issuing_id, Transmit_Send, transmit_info,
                    {node_id}, std::nullopt, std::nullopt,
                    0, 0, true, Trace({std::nullopt, transmit_id1}), true);
                this->IssueAction_ToCluster(recv, issuing_id, Transmit_AllocTensor, transmit_info,
                    {}, std::nullopt, Trace({std::nullopt, transmit_id2}),
                    1, 0, false, Trace({std::nullopt, transmit_id1}), true);
                this->IssueAction_ToCluster(recv, issuing_id, Transmit_Recv, transmit_info,
                    {}, node_id, std::nullopt,
                    0, 0, true, Trace({std::nullopt, transmit_id2}), true, transmit_id1);
            } else {
                this->IssueAction_ToCluster(send, issuing_id, Transmit_Send, transmit_info,
                    {node_id}, std::nullopt, std::nullopt,
                    1, 0, true, std::nullopt, true);
                this->IssueAction_ToCluster(recv, issuing_id, Transmit_Recv, transmit_info,
                    {}, node_id, std::nullopt,
                    1, 0, true, Trace({std::nullopt, transmit_id1}), true);
            }

        };
        this->transmit_manager->__register_callback_for_new_transmit(transmit_issuing);

        auto start_transmit = [this](std::shared_ptr<struct TransmitInfo> transmit_info) {
            this->PermitRecv_InCluster(transmit_info->recv_rank,
                                       transmit_info->transmit_id,
                                       transmit_info->variable_descriptor,
                                       transmit_info->issuing_id);
        };
        this->transmit_manager->__register_callback_for_transmit_permit(start_transmit);

        auto issuing_id_discard = [this](IssuingID issuing_id) {
            std::lock_guard<std::mutex> issuing_id__lock(this->issuing_id__lock);
            this->issuing_ids.releaseID(issuing_id);
        };
        this->GCG->__register_callback_for_issuing_id_discard(issuing_id_discard);

        auto checkpoint_placement_change = [this](NodeID ckpt_id, Rank location, bool add) {
            auto *ckpt = this->GCG->__get_node(ckpt_id);
            if (ckpt->_.is_scheduled || ckpt->_.done) {
                if (add) {
                    assert(0 < this->GCG->checkpoint_prelocations.at(ckpt_id).size());
                    ComputeDomain location_domain = this->rank_manager->GetComputeDomain(location);
                    Rank src = -1;
                    for (Rank r : this->GCG->checkpoint_prelocations.at(ckpt_id)) {
                        ComputeDomain src_domain = this->rank_manager->GetComputeDomain(r);
                        if (src_domain == location_domain) {
                            src = r;
                            break;
                        }
                    }
                    if (src == -1)
                        src = *(this->GCG->checkpoint_prelocations.at(ckpt_id).begin());

                    IssuingID issuing_id = -1; // TODO

                    this->IssueAction_ToCluster(src, issuing_id, FetchCheckpoint, std::make_shared<AccActionParamPayload>(),
                        {}, ckpt_id, std::nullopt,
                        1, 0, true, Trace({ckpt_id, std::nullopt}), false);
                    
                    HostID send_host = this->rank_manager->GetHostByRank(src);
                    std::string send_resource = this->rank_manager->GetResourceByHost(send_host);
                    ComputeDomain send_domain = this->rank_manager->GetComputeDomain(src);

                    HostID recv_host = this->rank_manager->GetHostByRank(location);
                    std::string recv_resource = this->rank_manager->GetResourceByHost(recv_host);
                    ComputeDomain recv_domain = this->rank_manager->GetComputeDomain(location);

                    this->transmit_manager->NewTransmitPair(src, send_resource, send_domain,
                        location, recv_resource, recv_domain,
                        ckpt_id,
                        true,
                        issuing_id);

                    auto action_param = std::make_shared<struct SettledAsCheckpoint>();
                    action_param->node_id = ckpt_id;
                    this->IssueAction_ToCluster(location, issuing_id, SettledAsCheckpoint, action_param,
                        {ckpt_id}, std::nullopt, Trace({ckpt_id, std::nullopt}),
                        1, 0, true, std::nullopt, false);
                } else {
                    this->FreeCheckpoint(location, ckpt_id);
                }
            }
        };
        this->GCG->__register_callback__for__checkpoint_placement_change(checkpoint_placement_change);

#ifndef NDEBUG
        auto debug_transmit = [this](std::shared_ptr<struct TransmitInfo> transmit_info) {
            if (!this->__display_more)
                return;
            std::cout << "PermitTransmit[" << transmit_info->transmit_id << "]" << std::endl;
        };
        this->transmit_manager->__register_callback_for_transmit_permit(debug_transmit);

        auto debug_op_insertion = [this](struct Node *node) {
        };
        this->GCG->__register_callback_for_node_insertion(Node_For_Op, debug_op_insertion);
        this->GCG->__register_callback_for_node_insertion(Node_For_Future, debug_op_insertion);
        this->GCG->__register_callback_for_node_insertion(Node_For_Task, debug_op_insertion);

        auto debug_execute_state_transfer = [this](struct Node *node,
                                                   bool old_scheduled, bool old_done,
                                                   bool new_scheduled, bool new_done) {
            if (!this->__display_more)
                return;
            if (old_scheduled == false && old_done == false && new_scheduled == true && new_done == false) {
                nlohmann::json jsoned_node = *node;
                jsoned_node.at("_").erase("shape");
                jsoned_node.at("_").erase("const_payload");
                std::cout << "Scheduling:" << jsoned_node.dump() << std::endl;
            }
        };
        this->GCG->__register_callback_for_execute_state_transfer(Node_For_Op, debug_execute_state_transfer);
        this->GCG->__register_callback_for_execute_state_transfer(Node_For_Future, debug_execute_state_transfer);
        this->GCG->__register_callback_for_execute_state_transfer(Node_For_Task, debug_execute_state_transfer);

        auto debug_storage_state_transfer = [this](struct Node *node,
                                                   bool old_checkpoint, bool new_checkpoint) {
        };
        this->GCG->__register_callback_for_storage_state_transfer(Node_For_Op, debug_storage_state_transfer);
        this->GCG->__register_callback_for_storage_state_transfer(Node_For_Future, debug_storage_state_transfer);
        this->GCG->__register_callback_for_storage_state_transfer(Node_For_Task, debug_storage_state_transfer);

        auto debug_op_deletion = [this](struct Node *node) {
            if (!this->__display_more)
                return;
            nlohmann::json j = *node;
            std::cout << "Deleting[" << node->node_id << "] ";
            std::cout << j.dump() << " " << std::endl;
        };
        this->GCG->__register_callback_for_node_deletion(Node_For_Op, debug_op_deletion);
        this->GCG->__register_callback_for_node_deletion(Node_For_Future, debug_op_deletion);
        this->GCG->__register_callback_for_node_deletion(Node_For_Task, debug_op_deletion);
#endif
    }
public:
    virtual void
    StartUp() override {
    }

    virtual void
    SendWatchdogEvent(MasterEventEnum event,
                      std::shared_ptr<MasterEventParamPayload> param_) override {
        if (event == WatchdogSignAccIn) {
            auto param = std::static_pointer_cast<struct WatchdogSignAccIn>(param_);
            std::unique_lock<std::shared_mutex> rank_lock(this->rank_manager__lock);

            this->rank_manager->sign_in(param);
            this->GCG->AccSignIn(param->rank);

            auto op_param = std::make_shared<struct HelloWorld>();
            op_param->i = param->rank;

            this->ActionStart();
            this->IssueAction_ToCluster(param->rank, std::nullopt, HelloWorld, op_param,
                {}, std::nullopt, std::nullopt,
                0, 0, true, std::nullopt, false);
            for (StreamID stream_id = 0; stream_id < NR_STREAM; stream_id++) {
                auto init_param = std::make_shared<struct InitATenRuntime>();
                init_param->stream_id = stream_id;
                this->IssueAction_ToCluster(param->rank, std::nullopt, InitATenRuntime, init_param,
                    {}, std::nullopt, std::nullopt,
                    stream_id, 0, true, std::nullopt, false);
            }
            this->ActionCommit();
        } else if (event == WatchdogSignAccOut) {
            assert(0);
        }
    }

    virtual void
    SendAccEvent(Rank from,
                 std::optional<IssuingID> issuing_id_,
                 Timestamp acc_timestamp,
                 MasterEventEnum event,
                 std::shared_ptr<MasterEventParamPayload> param_) override {

        if (event == AccReportNodeDone) {

            if (issuing_id_.has_value()) {
                IssuingID issuing_id = issuing_id_.value();
                std::lock_guard<std::mutex> issuing_id__lock(this->issuing_id__lock);
                if (!this->issuing_ids.contains(issuing_id))
                    return;
            }

            auto param = std::static_pointer_cast<struct AccReportNodeDone>(param_);
            std::shared_lock<std::shared_mutex> rank_lock(this->rank_manager__lock);
            bool debug_output;
            {
                std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);
                this->ActionStart();
                debug_output = this->GCG->__get_node(param->node_id)->_.debug;

                // 在这里也有个很麻烦的事
                // ReportCKPTArrived不见得已经report op done，因为ReportCKPTArrived事件可能先于 report op done
                // 就导致这里可能先mark node done，然后去掉所有前驱，并删掉这个node
                // 然后report op done的时候又来一次mark node done，找不到node导致报错
                this->GCG->MarkOpDone(param->node_id, 
                                      this->rank_manager->GetModelByHost(this->rank_manager->GetHostByRank(from)),
                                      param->running_time);
            }
            this->ActionCommit();

            if (this->__display_more
#ifdef NDEBUG
                && debug_output
#endif
            ) {
                std::ostringstream os;
                os << "report_op_done! "
                   << "rank[" << from << "] "
                   << "debug_output[" << debug_output << "] "
                   << "issuing_id[" << issuing_id_.value_or(0) << "] "
                   << "node_id[" << param->node_id << "] "
                   << "task_id[" << param->task_id << "] "
                   << "task_node_id[" << param->task_node_id << "] "
                   << "acc_model[" << this->rank_manager->GetModelByHost(this->rank_manager->GetHostByRank(from)) << "] "
                   << "end_timestamp[" << param->end << "] "
                   << "time(ns)[" << param->running_time << "]"
                   ;
                std::cout << os.str() << std::endl;
            }

            this->__schedule_one_batch();

        } else if (event == AccReportNodeFail) {
            auto param = std::static_pointer_cast<struct AccReportNodeFail>(param_);
            assert(0);

            this->__schedule_one_batch();
        } else if (event == AccRequestSend) {
            auto param = std::static_pointer_cast<struct AccRequestSend>(param_);
            std::shared_lock<std::shared_mutex> rank_lock(this->rank_manager__lock);
            {
                std::lock_guard<std::mutex> tran_lock(this->transmit_lock);
                this->ActionStart();
                bool ret = this->transmit_manager->RequestTransmitStart(param->transmit_id,
                                                                        param->variable_descriptor);
                assert(ret);
            }
            this->ActionCommit();
        } else if (event == AccReportRecvDone) {
            auto param = std::static_pointer_cast<struct AccReportRecvDone>(param_);
            Rank send;
            NodeID node_id;
            std::shared_lock<std::shared_mutex> rank_lock(this->rank_manager__lock);
            {
                std::lock_guard<std::mutex> tran_lock(this->transmit_lock);
                this->ActionStart();
                bool ret = this->transmit_manager->ReportTransmitDone(param->transmit_id, param);
                assert(ret);
                auto transmit_info = this->transmit_manager->transmit_id__to__info.at(param->transmit_id);
                send = transmit_info->send_rank;
                node_id = transmit_info->node__to_transmit;
                if (this->train_transmit_predictor) {
                    for (auto &p: param->profiling)
                        this->rank_manager->TrainTransmit(transmit_info->send_rank, transmit_info->recv_rank,
                                                          p.first, p.second);
                }
            }
            this->ActionCommit();

            // if (this->__display_more) {
            //     Duration duration = 0;
            //     NBytes nbytes = 0;
            //     for (auto &[bytes, time]: param->profiling) {
            //         duration += time;
            //         nbytes += bytes;
            //     }
            //     std::ostringstream os;
            //     os << "transmit_done! "
            //        << "node_id[" << node_id << "] "
            //        << "send[" << send << "] "
            //        << "recv[" << from << "] "
            //        << "end_timestamp[" << acc_timestamp << "] "
            //        << "nbytes[" << nbytes << "] "
            //        << "time(ns)[" << duration << "]"
            //        ;
            //     std::cout << os.str() << std::endl;
            // }

        } else if (event == AccReportCheckpointSettled) {
            auto param = std::static_pointer_cast<struct AccReportCheckpointSettled>(param_);
            std::shared_lock<std::shared_mutex> rank_lock(this->rank_manager__lock);
            std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);

            // 这里的node可能已经删去了，所以不能在这里this->GCG->__get_node(param->node_id)

            bool debug_output;
#ifndef NDEBUG
            debug_output = true;
#else
            debug_output = false;
#endif

            this->ActionStart();
            
            if (this->__display_more && debug_output)
                std::cout << "CKPT[" << param->node_id << "] "
                          << "Arrive Rank[" << from << "] "
                          << (this->rank_manager->IsReliable(from) ? "Reliable" : "") << std::endl;

            bool ret = this->GCG->ReportCheckpointArrived(from,
                                                          param->node_id,
                                                          this->rank_manager->IsReliable(from));
            if (!ret) {
                // 集群状态和GCG记录的不一致，取消掉这个checkpoint
                // 比如某个mark node done把一个checkpoint取消了，已经在GCG中删掉了
                // 但是SettleCheckpointDone还在集群中，就会触发这个。
                this->FreeCheckpoint(from, param->node_id);
            }
            this->ActionCommit();
        } else
            assert(0);


#ifdef TEST_SIMULATOR
        if (this->schedule_more) {
            // 在这里跑一套调度时模拟器，为了在每个master状态下都跑一遍调度时模拟器，测试模拟器代码别跑不通了
            // 提高模拟器代码的分支覆盖率

            std::lock_guard<std::mutex> issuing_id__lock(this->issuing_id__lock); // 这是个互斥锁

            Timestamp s, e;
            std::shared_ptr<Master> dumped_master_for_simulation;
            double dump_master_time, construct_time, simulating_time;

            {
                std::shared_lock<std::shared_mutex> rank_lock(this->rank_manager__lock);
                std::lock_guard<std::mutex> tran_lock(transmit_lock);
                std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);
                s = RealTimeNow();
                dumped_master_for_simulation = this->_dump_for_scheduling_time_simulation();
                e = RealTimeNow();
                dump_master_time = ((double)(e - s) / 1000000);
            }

            size_t nr_ops_simulated = 0;

            s = RealTimeNow();
            auto cloned_master = std::static_pointer_cast<SimpleFullMaster>(
                dumped_master_for_simulation->_clone());
            auto simulator = this->get_scheduletime_simulator(cloned_master);

            auto callback_for_action_done = [&](Rank r, std::shared_ptr<const AccActionSpec> a,
                                                Timestamp end, Duration duration) {
                nr_ops_simulated++;
            };
            simulator->__register_action_done_callback(callback_for_action_done);
            e = RealTimeNow();
            construct_time = ((double)(e - s) / 1000000);


            s = RealTimeNow();
            // 然后进行模拟
            while (simulator->Step() != EmptyAction)
                ;
            e = RealTimeNow();
            simulating_time = ((double)(e - s) / 1000000);

            std::ostringstream os;
            os << "TEST_SIMULATOR nr_ops_simulated[" << nr_ops_simulated << "] "
               << "dump_master_time[" << dump_master_time << "] "
               << "construct_time[" << construct_time << "] "
               << "simulating_time[" << simulating_time << "] ";
            std::cout << os.str() << std::endl;
        }
#endif
    }

private:
    std::shared_ptr<Simulator>
    get_scheduletime_simulator(
        std::shared_ptr<SimpleFullMaster> cloned_master) const {
        auto cluster_status = std::make_shared<ClusterStatusImpl>(cloned_master->rank_manager);
        auto _task_predictor = [this](AccModel acc_model, TaskID task_id, TaskNodeID task_node_id) -> Duration {
            return this->GCG->PredictTask(task_id, task_node_id, acc_model);
        };
        cluster_status->__register_task_predictor(_task_predictor);
    
        auto _transmit_predictor = [this](Rank s, Rank r, std::shared_ptr<VariableDescriptor> vd) -> Duration {
            assert(vd);
            Duration ret = 0;
            auto predict_one = [&] (std::shared_ptr<VariableDescriptor> &_v) {
                if (!_v->is_tensor)
                    return;
                ret += this->rank_manager->PredictTransmit(s, r,
                    _v->numel * c10::elementSize((c10::ScalarType)_v->dtype));
            };
            for_each_elem__in_tuple_vd(vd, predict_one);
            return ret;
        };
        cluster_status->__register_transmit_predictor(_transmit_predictor);

        auto _ask_node_shape = [=](NodeID node_id) -> std::shared_ptr<VariableDescriptor> {
            auto *node = cloned_master->GCG->__get_node(node_id);
            if (node->_.is_constant)
                return nullptr;
            return node->_.shape;
        };
        cluster_status->__register_ask_node_shape(_ask_node_shape);

        cluster_status->SetMaster(cloned_master);

        cluster_status->Init_Stage_2(cloned_master->Now(),
                                     *(cloned_master->transmit_manager),
                                     *(cloned_master->GCG));

        // 调度时模拟ClusterStatus中的Predictor与Rankmanager，复用真的Master
        // 所以ClusterStatus中的假数据可以影响不到Master，而且调度时模拟不会再触发动态调度
        cloned_master->GCG->SetTrainTaskPredictor(false);
        cloned_master->SetTrainTransmitPredictor(false);
        cloned_master->SetScheduleMore(false);
        cloned_master->DisplayMore(false);

        auto simulator = std::make_shared<Simulator>(cluster_status, cloned_master);
        cluster_status->Init_Stage_3(*(cloned_master->transmit_manager), *(cloned_master->GCG));
        return simulator;
    }

    void
    __schedule_one__static() { 
        std::lock_guard<std::mutex> __scheduling__lock(this->scheduling__lock);
        std::shared_lock<std::shared_mutex> rank_lock(this->rank_manager__lock);

        std::vector<NodeID> nodes_to_schedule;
        std::unordered_set<NodeID> ops;
        {
            std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);
            std::tie(nodes_to_schedule, ops) = this->GCG->_Step_1_SelectOpsToSchedule();
            if (nodes_to_schedule.empty())
                return;
            this->GCG->_Step_2_UpdateVertexCut(nodes_to_schedule, ops);
        }

        IssuingID issuing_id;
        {
            std::lock_guard<std::mutex> issuing_id__lock(this->issuing_id__lock);
            issuing_id = this->issuing_ids.allocateID();
        }

        std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);

        this->__issuing_ops(nodes_to_schedule, ops, issuing_id);
    }

    void
    __schedule_one_batch() {
        if (!this->schedule_more)
            return;

#ifdef STATIC_SCHEDULE
        this->__schedule_one__static();
        return;
#endif

        std::lock_guard<std::mutex> __scheduling__lock(this->scheduling__lock);
        std::shared_lock<std::shared_mutex> rank_lock(this->rank_manager__lock);

        std::vector<NodeID> nodes_to_schedule;
        std::unordered_set<NodeID> ops;
        {
            std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);
            std::tie(nodes_to_schedule, ops) = this->GCG->_Step_1_SelectOpsToSchedule();
            if (nodes_to_schedule.empty())
                return;
            this->GCG->_Step_2_UpdateVertexCut(nodes_to_schedule, ops);
        }

        IssuingID issuing_id;
        {
            std::lock_guard<std::mutex> issuing_id__lock(this->issuing_id__lock);
            issuing_id = this->issuing_ids.allocateID();
        }

        Timestamp s, e;
        std::shared_ptr<Master> dumped_master_for_simulation;
        {
            std::lock_guard<std::mutex> tran_lock(transmit_lock);
            std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);
            s = RealTimeNow();
            dumped_master_for_simulation = this->_dump_for_scheduling_time_simulation();
            e = RealTimeNow();
        }

        auto dumped_master = std::static_pointer_cast<SimpleFullMaster>(dumped_master_for_simulation);
        std::shared_ptr<GCG_Adding_OpManagement> example_GCG = dumped_master->GCG;
        std::shared_ptr<RankManager> rank_manager = dumped_master->rank_manager;


        auto _task_predictor = [this](AccModel acc_model, TaskID task_id, TaskNodeID task_node_id) -> Duration {
            return this->GCG->PredictTask(task_id, task_node_id, acc_model);
        };

        auto _transmit_predictor = [&](Rank from_rank, Rank to_rank, 
                                         std::shared_ptr<VariableDescriptor> shape) -> std::pair<Duration, NBytes> {
            if (from_rank == to_rank)
                return {0, 0};
        
            Duration transmit_duration = 0;
            NBytes transmit_size = 0;
            auto predict_one = [&] (std::shared_ptr<VariableDescriptor> &_v) {
                if (!_v->is_tensor)
                    return;
                NBytes tensor_size = _v->numel * c10::elementSize((c10::ScalarType)_v->dtype);
                transmit_size += tensor_size;
                transmit_duration += this->rank_manager->PredictTransmit(from_rank, to_rank, tensor_size);
            };
            for_each_elem__in_tuple_vd(shape, predict_one);
            return {transmit_duration, transmit_size};
        };

        auto get_scheduletime_simulator = [&]() -> std::shared_ptr<Simulator> {
            auto cloned_master = std::static_pointer_cast<SimpleFullMaster>(
                dumped_master_for_simulation->_clone());
            return this->get_scheduletime_simulator(cloned_master);
        };

        /*******************************************/
        /* 在这里跑调度算法 */
        std::unordered_map<NodeID, Rank> scheduling_results = this->GCG->_Step_3_Op_Scheduling(
            _task_predictor, _transmit_predictor,
            example_GCG, rank_manager,
            get_scheduletime_simulator,
            nodes_to_schedule
        );
        // 在这里调度算法跑完了，把调度结论写进当前master中

        std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);

        for (auto [node_id, rank]: scheduling_results) {
            auto *node = this->GCG->__get_node(node_id);
            this->__inject_assigned_ranks__for_ops(node_id, rank);
        }

        this->__issuing_ops(nodes_to_schedule, ops, issuing_id);
    }

    virtual void
    __inject_assigned_ranks__for_ops(NodeID node_id, Rank rank) {
        this->GCG->_Step_4_InjectAssigned_For_OPs(node_id, rank);
    }

    virtual void
    __issuing_ops(const std::vector<NodeID> &nodes_to_schedule,
                  const std::unordered_set<NodeID> &ops,
                  IssuingID issuing_id) {
        this->ActionStart();

        std::unordered_map<RankPair, std::unordered_set<NodeID>> issued_transmit;
        auto add_transmit =
            [&, this](Rank send, Rank recv, NodeID node_to_transmit, bool recv_for_settled_ckpt) {

            if (!issued_transmit.contains({send, recv}))
                issued_transmit.insert({{send, recv}, {}});

            if (issued_transmit.at({send, recv}).contains(node_to_transmit))
                return;
            
            HostID send_host = this->rank_manager->GetHostByRank(send);
            std::string send_resource = this->rank_manager->GetResourceByHost(send_host);
            ComputeDomain send_domain = this->rank_manager->GetComputeDomain(send);

            HostID recv_host = this->rank_manager->GetHostByRank(recv);
            std::string recv_resource = this->rank_manager->GetResourceByHost(recv_host);
            ComputeDomain recv_domain = this->rank_manager->GetComputeDomain(recv);
            this->transmit_manager->NewTransmitPair(send, send_resource, send_domain,
                recv, recv_resource, recv_domain,
                node_to_transmit,
                recv_for_settled_ckpt,
                issuing_id);
            issued_transmit.at({send, recv}).insert(node_to_transmit);
        };

        std::unordered_map<Rank, std::unordered_set<NodeID>> issued_fetch_ckpt;
        auto add_fetch_ckpt =
            [&, this](Rank loc, NodeID ckpt_id) {
            if (!issued_fetch_ckpt.contains(loc))
                issued_fetch_ckpt.insert({loc, {}});

            if (issued_fetch_ckpt.at(loc).contains(ckpt_id))
                return;

            this->IssueAction_ToCluster(loc, issuing_id, FetchCheckpoint, std::make_shared<AccActionParamPayload>(),
                {}, ckpt_id, std::nullopt,
                1, 0, true, Trace({ckpt_id, std::nullopt}), false);
            issued_fetch_ckpt.at(loc).insert(ckpt_id);
        };

        // 然后布置checkpoint的位置
        for (NodeID node_id: nodes_to_schedule) {
            auto *node = this->GCG->__get_node(node_id);
            node->_.checkpoint_src.clear();

            // 这个正在调度的算子，prelocation最初为空，所以这里向checkpoint中加入第一个prelocation
            if (node->_.is_checkpoint)
                this->GCG->__place_checkpoint_to_rank(node_id, node->_.assigned_to);

            for (NodeID input_id: node->inputs) {
                auto *input = this->GCG->__get_node(input_id);
                if (input->_.is_checkpoint) {
                    assert(!this->GCG->checkpoint_prelocations.at(input_id).empty());
#if 0   
                    // 不把checkpoint带到assigned_to上，只是把ckpt传过来计算，算完就扔掉
                    if (this->GCG->checkpoint_prelocations.at(input_id).contains(node->_.assigned_to))
                        node->_.checkpoint_src[input_id] = node->_.assigned_to;
                    else
                        node->_.checkpoint_src[input_id] = *(this->GCG->checkpoint_prelocations.at(input_id).begin());
#else   
                    // 把checkpoint带到assigned_to上
                    if (!this->GCG->checkpoint_prelocations.at(input_id).contains(node->_.assigned_to)) {
                        this->GCG->__place_checkpoint_to_rank(input_id, node->_.assigned_to);
                    }
                    node->_.checkpoint_src[input_id] = node->_.assigned_to;
#endif  
                }
            }

        }

        // 发射算子
        for (auto it = nodes_to_schedule.rbegin(); it != nodes_to_schedule.rend(); it++) {
            NodeID node_id = *it;
            auto *node = this->GCG->__get_node(node_id);
            Rank assigned_to = node->_.assigned_to;
            TaskID task_id = node->_.task_id;
            TaskNodeID task_node_id = node->_.task_node_id;
            size_t execution_sequence = node->_.execution_sequence;

            if (node->_.is_constant) {
                continue;
            } else if (node->_.has_tensor_payload) {
                auto param1 = std::make_shared<struct UploadTensor>();
                param1->f = node->_.const_payload.serialized_data;
                param1->node_id = node_id;
                this->IssueAction_ToCluster(assigned_to, issuing_id, UploadTensor, param1,
                    {}, node_id, std::nullopt,
                    0, 0, false, std::nullopt, false);
            } else /* if (!node->_.is_constant) */ {
                assert(!node->_.lost_inputs);
        
                // For operator output transmit-----------------------------------------------------------------
                std::unordered_set<Rank> node_user_ranks;
                for (auto user_node_id: node->uses) {
                    auto *user_node = this->GCG->__get_node(user_node_id);
                    if (ops.contains(user_node_id))
                        node_user_ranks.insert(user_node->_.assigned_to);
                }
                for (auto user_rank: node_user_ranks) {
                    if (user_rank != assigned_to)
                        add_transmit(assigned_to, user_rank, node_id, false);
                }

                // For the OP itself---------------------------------------------------------------
                auto action_param = std::make_shared<struct RunATenOP>();

                if (0 == node->_.target.size()) {
                    action_param->qualified_name = node->_.aten_op_name;
                } else {
                    action_param->qualified_name = "prim::GCG_Call_Submod";
                    if (this->GCG->task_manager)
                        // 新增Unscheduled算子、发射带有具体内容的graph，需要task_manager
                        // 全集群模拟、还有操纵真正集群运行的master需要task_manager
                        // 当GCG->task_manager为空时，说明是调度时模拟，不需要task_manager
                        action_param->graph = std::get<1>(this->GCG->task_manager->tasks.at(task_id)).at(node->_.target);
                }
                action_param->node_id = node_id;
                action_param->task_id = task_id;
                action_param->task_node_id = task_node_id;
                action_param->nr_op_inputs = node->inputs.size();
                action_param->op_input__to__action_input.resize(action_param->nr_op_inputs);
                action_param->op_input__to__consts.resize(action_param->nr_op_inputs);

                std::vector<NodeID> action_input_ids;
                for (size_t input_idx = 0; input_idx < action_param->nr_op_inputs; input_idx++) {
                    NodeID node_input_id = node->inputs[input_idx];
        
                    action_param->op_input__to__consts[input_idx] = -1;
                    action_param->op_input__to__action_input[input_idx] = -1;

                    auto *input_node = this->GCG->__get_node(node_input_id);
        
                    if (input_node->_.is_constant) {
                        action_param->op_input__to__consts[input_idx] = action_param->constants.size();
                        action_param->constants.push_back(input_node->_.const_payload);
                    } else {
                        action_param->op_input__to__action_input[input_idx] = action_input_ids.size();
                        action_input_ids.push_back(node_input_id);
                    }
                }

                this->IssueAction_ToCluster(assigned_to, issuing_id, RunATenOP, action_param,
                    action_input_ids, node_id, std::nullopt,
                    0, 0, false, std::nullopt, false, execution_sequence);
        
                // For checkpoint producer input-----------------------------------------------------------
                for (auto p: node->_.checkpoint_src) {
                    NodeID node_input_id = p.first;
                    Rank checkpoint_src = p.second;
                    if (checkpoint_src == assigned_to) {
                        add_fetch_ckpt(assigned_to, node_input_id);
                    } else {
                        add_transmit(checkpoint_src, assigned_to, node_input_id, false);
                        add_fetch_ckpt(checkpoint_src, node_input_id);
                    }
                }
            }

            if (node->_.is_checkpoint && !this->GCG->checkpoint_prelocations.at(node_id).empty()) {
                for (Rank ckpt_location: this->GCG->checkpoint_prelocations.at(node_id)) {
                    auto action_param = std::make_shared<struct SettledAsCheckpoint>();
                    action_param->node_id = node_id;
                    this->IssueAction_ToCluster(ckpt_location, issuing_id, SettledAsCheckpoint, action_param,
                        {node_id}, std::nullopt, Trace({node_id, std::nullopt}),
                        1, 0, true, std::nullopt, false);
                    if (assigned_to != ckpt_location) 
                        add_transmit(assigned_to, ckpt_location, node_id, true);
                }
            }
        }

        this->ActionCommit();
        this->GCG->_Step_5_PostIssuing(nodes_to_schedule, ops, issuing_id);
    }

    virtual std::vector<Future>
    RunTask(TaskID task_id,
            std::vector<Future> input_futures,
            std::optional<std::vector<Rank>> manual_assignment,
            std::vector<size_t> debug_output_i) override {
        std::vector<Future> ret;
        {
            std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);
            ret = this->GCG->InsertUnscheduledTask(task_id, input_futures, manual_assignment, debug_output_i);
        }
        this->__schedule_one_batch();
        return ret;
    }

    virtual Future
    UploadTensor_ToCluster(std::vector<char> serialized_data, std::vector<long int> shape, int dtype) override {
        Future ret;
        {
            std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);
            ret = this->GCG->InsertUnscheduledConstantTensor(serialized_data,
                                                             std::make_shared<VariableDescriptor>(shape, dtype));
        }
        this->__schedule_one_batch();
        return ret;
    }

    virtual void
    DropFuture(Future fut) override {
        std::shared_lock<std::shared_mutex> rank_lock(this->rank_manager__lock);
        {
            std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);
            this->ActionStart();
            this->GCG->DropFuture(fut);
        }
        this->ActionCommit();
    }

    virtual TaskID
    SubmitGraph(std::string root_graph,
                std::unordered_map<std::string, std::string> sub_graphs,
                std::unordered_map<int, std::string> symbol__to__symexpr,
                bool all_links_to_successor = false) override {
        std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);
        return this->GCG->SubmitGraph(root_graph, sub_graphs, symbol__to__symexpr, all_links_to_successor);
    }

    virtual void
    DropTask(TaskID task_id) override {
        {
            std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);
            this->ActionStart();
            this->GCG->DropTask(task_id);
        }
        this->ActionCommit();
    }

    virtual nlohmann::json
    ExportMasterStatus() override {
        std::lock_guard<std::mutex> __scheduling__lock(this->scheduling__lock);
        std::unique_lock<std::shared_mutex> rank_lock(this->rank_manager__lock);
        std::lock_guard<std::mutex> tran_lock(transmit_lock);
        std::lock_guard<std::mutex> gcg_lock(this->GCG__lock);
        std::lock_guard<std::mutex> issuing_id__lock(this->issuing_id__lock);
        return this->ToJson();
    }
};

std::shared_ptr<Master>
GetSimpleFullMaster() {
    return std::make_shared<SimpleFullMaster>();
}

std::shared_ptr<Master>
GetSimpleFullMasterFromJson(const nlohmann::json &j) {
    return std::make_shared<SimpleFullMaster>(j);
}
