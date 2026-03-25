#include "Base.h"

class SingleAccGraphMaster final: public Master {
public:
  virtual void StartUp() override { }

  virtual void SendWatchdogEvent(MasterEventEnum event,
          std::shared_ptr<MasterEventParamPayload> param_) override {
    if (event == WatchdogSignAccIn) {
      auto param = std::static_pointer_cast<struct WatchdogSignAccIn>(param_);
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

      std::string zero_tensor_graph =
"\
graph():\n\
  %12 : bool? = prim::Constant()\n\
  %10 : Device? = prim::Constant()\n\
  %6 : int? = prim::Constant()\n\
  %4 : int[] = prim::Constant[value=[3,3]]()\n\
  %rv.1 : Tensor = aten::zeros(%4, %6, %6, %10, %12)\n\
  return (%rv.1)\n\
";

      auto op1_param = std::make_shared<struct RunATenOP>();
      op1_param->qualified_name = "prim::GCG_Call_Submod";
      op1_param->node_id = 1;
      op1_param->task_id = 2;
      op1_param->task_node_id = 3;
      op1_param->graph = zero_tensor_graph;
      op1_param->nr_op_inputs = 0;

      auto op2_param = std::make_shared<struct RunATenOP>();
      op2_param->qualified_name = "prim::GCG_Call_Submod";
      op2_param->node_id = 4;
      op2_param->task_id = 5;
      op2_param->task_node_id = 6;
      op2_param->graph = zero_tensor_graph;
      op2_param->nr_op_inputs = 0;

      auto op3_param = std::make_shared<struct RunATenOP>();
      op3_param->qualified_name = "aten::mul";
      op3_param->node_id = 7;
      op3_param->task_id = 8;
      op3_param->task_node_id = 9;
      op3_param->nr_op_inputs = 2;
      op3_param->op_input__to__action_input = {0, 1};
      op3_param->op_input__to__consts = {-1, -1};

      this->ActionStart();
      this->IssueAction_ToCluster(param->rank, std::nullopt, RunATenOP, op1_param,
              {}, 1, std::nullopt,
              0, 0, false, std::nullopt, false);
      this->IssueAction_ToCluster(param->rank, std::nullopt, RunATenOP, op2_param,
              {}, 2, std::nullopt,
              0, 0, false, std::nullopt, false);
      this->IssueAction_ToCluster(param->rank, std::nullopt, RunATenOP, op3_param,
              {1, 2}, 3, std::nullopt,
              0, 0, false, std::nullopt, false);
      this->ActionCommit();

    }
  }

  virtual void SendAccEvent(Rank from,
          std::optional<IssuingID>,
          Timestamp acc_timestamp,
          MasterEventEnum,
          std::shared_ptr<MasterEventParamPayload>) override {
  }
};

std::shared_ptr<Master>
GetSingleAccGraphMaster() {
  return std::make_shared<SingleAccGraphMaster>();
}
