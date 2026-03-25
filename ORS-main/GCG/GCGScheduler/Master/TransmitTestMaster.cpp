#include "Base.h"
#include <unordered_set>

#include "Master/FullMaster/TransmitManager.h"
#include "Master/FullMaster/RankManager.h"

class TransmitTestMaster final: public Master {
public:
  TransmitTestMaster() {
    this->cur_node_id = 0;
    this->rank_manager = std::make_shared<RankManager>();
    this->transmit_manager = Get_SimpleTransmitManager();

    auto timer = [this]() {
        return this->Now();
    };
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

    auto debug_transmit = [this](std::shared_ptr<struct TransmitInfo> transmit_info) {
        std::cout << "PermitTransmit[" << transmit_info->transmit_id << "]" << std::endl;
    };
    this->transmit_manager->__register_callback_for_transmit_permit(debug_transmit);
  }
  virtual void StartUp() override { }

  virtual void SendWatchdogEvent(MasterEventEnum event,
          std::shared_ptr<MasterEventParamPayload> param_) override {
    if (event == WatchdogSignAccIn) {
      auto param = std::static_pointer_cast<struct WatchdogSignAccIn>(param_);

      std::unique_lock<std::shared_mutex> rank_lock(this->rank_manager__lock);
      std::lock_guard<std::mutex> tran_lock(transmit_lock);

      this->rank_manager->sign_in(param);

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

      auto get_node_id = [&] () {
        return (this->cur_node_id)++;
      };

      auto add_transmit =
          [&, this](Rank send, Rank recv, NodeID node_to_transmit) {
          HostID send_host = this->rank_manager->GetHostByRank(send);
          std::string send_resource = this->rank_manager->GetResourceByHost(send_host);
          ComputeDomain send_domain = this->rank_manager->GetComputeDomain(send);

          HostID recv_host = this->rank_manager->GetHostByRank(recv);
          std::string recv_resource = this->rank_manager->GetResourceByHost(recv_host);
          ComputeDomain recv_domain = this->rank_manager->GetComputeDomain(recv);
          this->transmit_manager->NewTransmitPair(send, send_resource, send_domain,
              recv, recv_resource, recv_domain,
              node_to_transmit,
              false,
              std::nullopt);
      };

      auto issuing_simple_ops = [&] (Rank r1, Rank r2) {

        std::string zero_tensor_graph =
"\
graph():\n\
  %12 : bool? = prim::Constant()\n\
  %10 : Device = prim::GCG_get_native_device()\n\
  %6 : int? = prim::Constant()\n\
  %4 : int[] = prim::Constant[value=[3,3]]()\n\
  %rv.1 : Tensor = aten::zeros(%4, %6, %6, %10, %12)\n\
  return (%rv.1)\n\
";

        auto op1_param = std::make_shared<struct RunATenOP>();
        op1_param->qualified_name = "prim::GCG_Call_Submod";
        op1_param->node_id = get_node_id();
        op1_param->task_id = 1;
        op1_param->task_node_id = 1;
        op1_param->graph = zero_tensor_graph;
        op1_param->nr_op_inputs = 0;

        auto op2_param = std::make_shared<struct RunATenOP>();
        op2_param->qualified_name = "prim::GCG_Call_Submod";
        op2_param->node_id = get_node_id();
        op2_param->task_id = 1;
        op2_param->task_node_id = 2;
        op2_param->graph = zero_tensor_graph;
        op2_param->nr_op_inputs = 0;

        auto op3_param = std::make_shared<struct RunATenOP>();
        op3_param->qualified_name = "aten::mul";
        op3_param->node_id = get_node_id();
        op3_param->task_id = 1;
        op3_param->task_node_id = 3;
        op3_param->nr_op_inputs = 2;
        op3_param->op_input__to__action_input = {0, 1};
        op3_param->op_input__to__consts = {-1, -1};

        this->IssueAction_ToCluster(r1, std::nullopt, RunATenOP, op1_param,
                {}, op1_param->node_id, std::nullopt,
                0, 0, false, std::nullopt, false);
        this->IssueAction_ToCluster(r1, std::nullopt, RunATenOP, op2_param,
                {}, op2_param->node_id, std::nullopt,
                0, 0, false, std::nullopt, false);

        add_transmit(r1, r2, op1_param->node_id);
        add_transmit(r1, r2, op2_param->node_id);

        this->IssueAction_ToCluster(r2, std::nullopt, RunATenOP, op3_param,
                {op1_param->node_id, op2_param->node_id}, op3_param->node_id, std::nullopt,
                0, 0, false, std::nullopt, false);
      };

      auto ranks = this->rank_manager->GetAllRanks();
      if (2 <= ranks.size()) {
        this->ActionStart();
        issuing_simple_ops(ranks[ranks.size() - 2], ranks[ranks.size() - 1]);
        this->ActionCommit();
      }
    }
  }

  virtual void SendAccEvent(Rank from,
          std::optional<IssuingID> issuing_id_,
          Timestamp acc_timestamp,
          MasterEventEnum event,
          std::shared_ptr<MasterEventParamPayload> param_) override {
    if (event == AccRequestSend) {
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
    } else if (event == AccReportNodeDone) {
        auto param = std::static_pointer_cast<struct AccReportNodeDone>(param_);
        std::ostringstream os;
        os << "report_op_done! "
           << "rank[" << from << "] "
           << "issuing_id[" << issuing_id_.value_or(0) << "] "
           << "node_id[" << param->node_id << "] "
           << "task_id[" << param->task_id << "] "
           << "task_node_id[" << param->task_node_id << "] "
           << "acc_model[" << this->rank_manager->GetModelByHost(this->rank_manager->GetHostByRank(from)) << "] "
           << "end_timestamp[" << param->end << "] "
           << "time(ns)[" << param->running_time << "]";
        std::cout << os.str() << std::endl;
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
        for (auto &p: param->profiling) {
            this->rank_manager->TrainTransmit(transmit_info->send_rank, transmit_info->recv_rank,
                                              p.first, p.second);
        }
      }
      this->ActionCommit();

      Duration duration = 0;
      NBytes nbytes = 0;
      for (auto &[bytes, time]: param->profiling) {
          duration += time;
          nbytes += bytes;
      }
      std::ostringstream os;
      os << "transmit_done! "
         << "node_id[" << node_id << "] "
         << "send[" << send << "] "
         << "recv[" << from << "] "
         << "end_timestamp[" << acc_timestamp << "] "
         << "nbytes[" << nbytes << "] "
         << "time(ns)[" << duration << "]"
         << std::endl;
      std::cout << os.str();
    }
  }
private:
  std::shared_mutex rank_manager__lock;
  std::shared_ptr<RankManager> rank_manager;

  std::mutex transmit_lock;
  std::shared_ptr<TransmitManager> transmit_manager;

  TransmitID cur_transmit_id;
  NodeID cur_node_id;
};

std::shared_ptr<Master>
GetTransmitTestMaster() {
  return std::make_shared<TransmitTestMaster>();
}
