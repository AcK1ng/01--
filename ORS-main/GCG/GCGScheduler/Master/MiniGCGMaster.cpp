#include "Base.h"
#include "FullMaster/RankManager.h"

#include <map>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <mutex>
#include <algorithm>
#include <list>

#include <torch/jit.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/constants.h>

using TSGraph = torch::jit::Graph;
using TSGraph_ptr = std::shared_ptr<TSGraph>;


struct WorkerPair{
  Rank sender;
  Rank receiver;
  WorkerPair(Rank s, Rank r) : sender(s), receiver(r) {}  

};

class MiniGCGMaster final: public Master {
public:
  MiniGCGMaster() {
    this->last_rank__sign_in = -1;
    this->cur_transmit_id = 0;
    
    this->cur_task_id = 0;
    this->cur_issuing_id = 0;
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    
    this->GCG.setInsertPoint(this->GCG.return_node());

  }
  virtual void StartUp() override { }

  virtual void SendWatchdogEvent(MasterEventEnum event,
          std::shared_ptr<MasterEventParamPayload> param_) override {
    this->lock.lock();
    if (event == WatchdogSignAccIn) {
      auto param = std::static_pointer_cast<struct WatchdogSignAccIn>(param_);

      
      rank_manager_.sign_in(
          param->rank,
          param->host_id,
          param->acc_model,
          param->resource,
          param->domain
      );
      
      
      rank_manager_.PrintTopologySummary();  

      this->ranks.insert({param->rank, param});

      auto op_param = std::make_shared<struct HelloWorld>();
      op_param->i = param->rank;

      this->ActionStart();
      this->IssueAction_ToCluster(param->rank, std::nullopt, HelloWorld, op_param,
          {}, std::nullopt, std::nullopt,
          0, 0, true, std::nullopt, false);
      for (StreamID stream_id = 0; stream_id < NR_STREAM; stream_id++)
        this->IssueAction_ToCluster(param->rank, std::nullopt, InitATenRuntime, std::make_shared<AccActionParamPayload>(),
            {}, std::nullopt, std::nullopt,
            stream_id, 0, true, std::nullopt, false);
      this->ActionCommit();
      this->last_rank__sign_in = param->rank;
    }
    this->lock.unlock();
  }

  bool _transmit_can_start(Rank sender_rank, Rank receiver_rank, std::shared_ptr<struct WorkerPair> worker_pair) {

    if (this->transmit_worker_pairs_to_busy_transmit.find(worker_pair) == this->transmit_worker_pairs_to_busy_transmit.end() &&  
        (this->transmit_busy_workers.find(sender_rank) == this->transmit_busy_workers.end() ||  this->transmit_busy_workers.find(receiver_rank) == this->transmit_busy_workers.end())) 
      return false;  

    return true;
  }

  virtual void SendAccEvent(Rank from,
          std::optional<IssuingID> issuing_id,
          Timestamp acc_timestamp,
          MasterEventEnum event,
          std::shared_ptr< MasterEventParamPayload> param_) override {
    this->lock.lock();

    if (event == AccRequestSend) 
    {
      auto param = std::static_pointer_cast<struct AccRequestSend>(param_);

      std::cout << "AccRequestSend TransmitID[" << param->transmit_id << "] "
                << "Sender[" << from << "] "
                << "Receiver[" << param->recv_rank << "] "
                << std::endl;
      
      std::shared_ptr<WorkerPair> worker_pair = std::make_shared<WorkerPair>(from, param->recv_rank);  

      this->transmit_id_to_info[param->transmit_id] = std::make_pair(from, param);
      if(this->_transmit_can_start(from, param->recv_rank, worker_pair)) 
      {
        this->ActionStart();
        this->PermitRecv_InCluster(param->recv_rank,
                                    param->transmit_id,
                                    param->variable_descriptor,
                                    issuing_id);
        this->ActionCommit();

        if (this->transmit_worker_pairs_to_busy_transmit.find(worker_pair) == this->transmit_worker_pairs_to_busy_transmit.end())
        {
          this->transmit_busy_workers.insert(from);
          this->transmit_busy_workers.insert(param->recv_rank);
        
          this->transmit_worker_pairs_to_busy_transmit[worker_pair] = std::set<TransmitID>();
        }
        this->transmit_worker_pairs_to_busy_transmit[worker_pair].insert(param->transmit_id);
      } 
      else 
      {
        if (this->transmit_worker_pairs_to_idle_transmit.find(worker_pair) == this->transmit_worker_pairs_to_idle_transmit.end())
        {
          this->transmit_worker_pairs_to_idle_transmit[worker_pair] = std::set<TransmitID>();
        }
        this->transmit_worker_pairs_to_idle_transmit[worker_pair].insert(param->transmit_id);
      }
    } 
    else if (event == AccReportRecvDone) 
    {
      auto param = std::static_pointer_cast<struct AccReportRecvDone>(param_);
      auto transmit_info = this->transmit_id_to_info[param->transmit_id];
      Rank sender_rank = transmit_info.first;
      Rank receiver_rank = transmit_info.second->recv_rank;
      // TODO:
      // self.transmit_performance_predictor.train(sender_rank, receiver_rank, size_time_pairs)
      this->transmit_id_to_info.erase(param->transmit_id);
      std::shared_ptr<WorkerPair> worker_pair = std::make_shared<WorkerPair>(sender_rank, receiver_rank);  

      this->transmit_worker_pairs_to_busy_transmit[worker_pair].erase(param->transmit_id);


      if(this->transmit_worker_pairs_to_busy_transmit[worker_pair].size() == 0)
      {
        this->transmit_worker_pairs_to_busy_transmit.erase(worker_pair);
        this->transmit_busy_workers.erase(sender_rank);
        this->transmit_busy_workers.erase(receiver_rank);
      }


      for (const auto& entry : transmit_worker_pairs_to_idle_transmit) 
      {  
        const auto& idle_worker_pair  = entry.first;  
        Rank sender = idle_worker_pair->sender;
        Rank receiver = idle_worker_pair->receiver;
        if (this->_transmit_can_start(sender, receiver, idle_worker_pair))
        {

          if (this->transmit_worker_pairs_to_busy_transmit.find(idle_worker_pair) == this->transmit_worker_pairs_to_busy_transmit.end())
          {
            this->transmit_busy_workers.insert(sender);
            this->transmit_busy_workers.insert(receiver);
            this->transmit_worker_pairs_to_busy_transmit[idle_worker_pair] = std::set<TransmitID>();
          }
          for(auto idle_transmit_id: entry.second)
          {
            auto transmit_info = this->transmit_id_to_info[idle_transmit_id];
            Rank sender = transmit_info.first;
            auto transmit_param = transmit_info.second; 
            this->ActionStart();
            this->PermitRecv_InCluster(transmit_param->recv_rank,
                                      transmit_param->transmit_id,
                                      transmit_param->variable_descriptor,
                                      issuing_id);
            this->ActionCommit();
            this->transmit_worker_pairs_to_busy_transmit[idle_worker_pair].insert(idle_transmit_id); 
          }
          this->transmit_worker_pairs_to_idle_transmit.erase(idle_worker_pair);
        }
      }
    } 
    else if (event == AccReportNodeDone) 
    {
      auto param = std::static_pointer_cast<struct AccReportNodeDone>(param_);

      std::cout << "report_op_done! "
                << "rank[" << from << "] "
                << "acc_name[" << this->ranks[from]->acc_name << "] "
                << "issuing_id[" << issuing_id.value_or(0) << "] "
                << "node_id[" << param->node_id << "] "
                << "task_id[" << param->task_id << "] "
                << "task_node_id[" << param->task_node_id << "] "
                << "acc_model[" << this->ranks[from]->acc_model << "] "
                << "end_timestamp[" << param->end << "] "
                << "time(ns)[" << param->running_time << "]"
                << std::endl;
    }
    this->lock.unlock();
  }
  

  virtual TaskID
  SubmitGraph(std::string root_graph,
          std::unordered_map<std::string, std::string> sub_graphs,
          std::unordered_map<int, std::string> symbol__to__symexpr,
          bool all_links_to_successor = false) override {
    TSGraph_ptr graph = std::make_shared<torch::jit::Graph>();
    torch::jit::parseIR(root_graph, graph.get());

    this->lock.lock();

    TaskID task_id = this->cur_task_id++;

    this->__fast_assign__no_lock(graph);

    TaskNodeID task_node_id = 0;
    for (auto node: graph->nodes()) {
      node->i_(torch::jit::Symbol::attr("taskId"), task_id);
      node->i_(torch::jit::Symbol::attr("taskNodeId"), task_node_id++);
    }

    this->tasks[task_id] = {graph, sub_graphs};

    rank_manager_.PrintTopologySummary(); 

    this->lock.unlock();
    return task_id;
  }

private:
  inline void
  __fast_assign__no_lock(TSGraph_ptr graph) {
    size_t nr_nodes = 0;
    for (auto node: graph->nodes())
      nr_nodes++;

    size_t nr_ranks = this->last_rank__sign_in + 1;

    size_t nr_nodes__per_rank = nr_nodes / nr_ranks;
    if (nr_nodes % nr_ranks != 0)
      nr_nodes__per_rank++;

    size_t node_idx = 0;
    for (auto node: graph->nodes()) {
      node->i_(torch::jit::Symbol::attr("assignedTo"),
              node_idx / nr_nodes__per_rank);
      node_idx++;
    }
  }

public:
  virtual std::vector<Future>
  RunTask(TaskID task_id,
          std::vector<Future> inputs,
          std::optional<std::vector<Rank>> manual_assignment,
          std::vector<size_t> debug_output_i) override {
    this->lock.lock();

    auto unscheduled_below = this->GCG.createNone();
    this->GCG.insertNode(unscheduled_below);
    auto input_values = std::vector<torch::jit::Value *>(inputs.size());
    std::transform(inputs.cbegin(), inputs.cend(),
            input_values.begin(),
            [&] (Future future) {
              NodeID node_id = future;
              return this->node_id__to__node[node_id]->output(0);
            });

    auto task_graph = this->tasks[task_id].first;
    
    auto output_values = torch::jit::insertGraph(this->GCG, *task_graph, input_values);

    auto outputs = std::vector<Future>(output_values.size());
    std::transform(output_values.cbegin(), output_values.cend(),
            outputs.begin(),
            [&] (torch::jit::Value *value) {
              return value->unique();
            });

    std::list<torch::jit::Node *> unscheduled_nodes;
    for (torch::jit::Node *node = this->GCG.return_node()->prev();
            node != unscheduled_below;
            node = node->prev())
      unscheduled_nodes.push_front(node);

    unscheduled_below->destroy();

    for (auto node: unscheduled_nodes)
      this->node_id__to__node[node->output(0)->unique()] = node;

#ifndef NDEBUG
    for (auto *node: unscheduled_nodes)
      std::cout << *node;
#endif

    this->__schedule_nodes__no_lock(task_id, inputs, unscheduled_nodes, outputs);

    this->lock.unlock();
    return outputs;
  }

  virtual Future
  UploadTensor_ToCluster(std::vector<char> f, std::vector<long int> shape, int dtype) override {
    this->lock.lock();

    auto upload_tensor = this->GCG.create(
            torch::jit::Symbol::fromQualString("GCG::tensor_uploaded"), 1);
    this->GCG.insertNode(upload_tensor);


    TaskID task_id = this->cur_task_id++;
    TaskNodeID task_node_id = 0;
    upload_tensor->i_(torch::jit::Symbol::attr("taskId"), task_id);
    upload_tensor->i_(torch::jit::Symbol::attr("taskNodeId"), task_node_id++);
    upload_tensor->i_(torch::jit::Symbol::attr("assignedTo"), 0);

    auto output = upload_tensor->output(0)->unique();

#ifndef NDEBUG
    std::cout << *upload_tensor << std::endl;
#endif

    Rank assigned_to = upload_tensor->i(torch::jit::Symbol::attr("assignedTo"));
    IssuingID issuing_id = ++this->cur_issuing_id;

    this->ActionStart();

    auto action_param1 = std::make_shared<struct UploadTensor>();
    action_param1->f = std::make_shared<std::vector<char>>(f);
    action_param1->node_id = output;
    this->IssueAction_ToCluster(assigned_to, issuing_id, UploadTensor, action_param1,
            {}, output, std::nullopt,
            0, 0, false, std::nullopt, false);

    auto action_param2 = std::make_shared<struct SettledAsCheckpoint>();
    action_param2->node_id = output;
    this->IssueAction_ToCluster(assigned_to, issuing_id, SettledAsCheckpoint, action_param2,
            {output}, std::nullopt, Trace({output, std::nullopt}),
            0, 0, true, std::nullopt, false);
    this->ActionCommit();
    
    this->node_id__to__node[output] = upload_tensor;
    this->checkpoint_locations[output].insert(assigned_to);

    this->lock.unlock();
    return output;
  }

private:
  inline void
  __schedule_nodes__no_lock(
          TaskID task_id,
          std::vector<Future> &inputs,
          std::list<torch::jit::Node *> &unscheduled_nodes,
          std::vector<Future> &outputs) {

    IssuingID issuing_id = ++this->cur_issuing_id;

    auto add_transmit = [&] (Rank send, Rank recv, NodeID node_id) {
      auto transmit_param = std::make_shared<struct TransmitInfo>();
      transmit_param->transmit_id = this->cur_transmit_id;
      transmit_param->node__to_transmit = node_id;
      transmit_param->send_resource = "cuda";
      transmit_param->send_rank = send;
      transmit_param->send_host = 0;
      transmit_param->recv_resource = "cuda";
      transmit_param->recv_rank = recv;
      transmit_param->recv_host = 0;

      this->IssueAction_ToCluster(send, issuing_id, Transmit_RequestSendToMaster, transmit_param,
              {node_id}, std::nullopt, std::nullopt,
              1, 0, true, std::nullopt, false);
      this->IssueAction_ToCluster(send, issuing_id, Transmit_Send, transmit_param,
              {node_id}, std::nullopt, std::nullopt,
              0, 0, true, Trace({std::nullopt, this->cur_transmit_id}), true);
      this->IssueAction_ToCluster(recv, issuing_id, Transmit_AllocTensor, transmit_param,
              {}, std::nullopt, Trace({std::nullopt, this->cur_transmit_id + 1}),
              1, 0, false, Trace({std::nullopt, this->cur_transmit_id}), true);
      this->IssueAction_ToCluster(recv, issuing_id, Transmit_Recv, transmit_param,
              {}, node_id, std::nullopt,
              0, 0, true, Trace({std::nullopt, this->cur_transmit_id + 1}), true);

      this->cur_transmit_id += 2;
    };

    std::unordered_map<NodeID, c10::IValue> id_to_const;
    for (auto it = unscheduled_nodes.cbegin(); it != unscheduled_nodes.cend(); it++) {
      auto node = *it;
      auto node_value = node->output(0);
      auto node_id = node_value->unique();
      std::optional<torch::jit::IValue> const_ival = torch::jit::toIValue(node_value);
      if (const_ival.has_value())
        id_to_const[node_id] = const_ival.value();
    }

    this->ActionStart();

    // For checkpoint output---------------------------------------------------------------
    for (NodeID output: outputs) {
      auto node = this->node_id__to__node[output];
      Rank node__assigned_to = node->i(torch::jit::Symbol::attr("assignedTo"));
      auto action_param = std::make_shared<struct SettledAsCheckpoint>();
      action_param->node_id = output;
      this->IssueAction_ToCluster(node__assigned_to, issuing_id, SettledAsCheckpoint, action_param,
              {output}, std::nullopt, Trace({output, std::nullopt}),
              1, 0, true, std::nullopt, false);
      this->checkpoint_locations.insert({output, {node__assigned_to}});
    }

    auto &sub_graphs_str = this->tasks[task_id].second;
    auto checkpoint_inputs = std::unordered_set<NodeID>(inputs.cbegin(), inputs.cend());

    for (auto it = unscheduled_nodes.crbegin(); it != unscheduled_nodes.crend(); it++) {
      auto node = *it;
      auto node_id = node->output(0)->unique();
      if (id_to_const.contains(node_id))
        continue;

      Rank assigned_to = node->i(torch::jit::Symbol::attr("assignedTo"));

      auto inputs_value = node->inputs();
      auto inputs_node_id = std::vector<NodeID>(inputs_value.size());
      std::transform(inputs_value.cbegin(), inputs_value.cend(),
              inputs_node_id.begin(),
              [&] (torch::jit::Value *value) {
              return value->unique();
              });

      // For transmit----------------------------------------------------------------------
      std::unordered_set<Rank> node_user_ranks;
      for (auto &use: node->output(0)->uses()) {
        auto user = use.user;
        node_user_ranks.insert(user->i(torch::jit::Symbol::attr("assignedTo")));
      }
      for (auto user_rank: node_user_ranks)
        if (user_rank != assigned_to)
          add_transmit(assigned_to, user_rank, node_id);

      // For the OP itself---------------------------------------------------------------
      auto action_param = std::make_shared<struct RunATenOP>();
      action_param->qualified_name = node->kind().toQualString();
      action_param->node_id = node_id;
      action_param->task_id = task_id;
      action_param->task_node_id = node->i(torch::jit::Symbol::attr("taskNodeId"));

      if (action_param->qualified_name == "prim::GCG_Call_Submod")
        action_param->graph = sub_graphs_str[node->s(torch::jit::Symbol::attr("target"))];

      action_param->nr_op_inputs = node->inputs().size();

      std::vector<NodeID> action_input_ids;

      const auto node_input_values = node->inputs();
      for (size_t input_idx = 0; input_idx < action_param->nr_op_inputs; input_idx++) {
        NodeID node_input_id = node_input_values[input_idx]->unique();

        action_param->op_input__to__consts.push_back(-1);
        action_param->op_input__to__action_input.push_back(-1);

        if (id_to_const.contains(node_input_id)) {
          action_param->op_input__to__consts[input_idx] = action_param->constants.size();
          action_param->constants.push_back(id_to_const[node_input_id]);
        } else {
          action_param->op_input__to__action_input[input_idx] = action_input_ids.size();
          action_input_ids.push_back(node_input_id);
        }
        
      }

      this->IssueAction_ToCluster(assigned_to, issuing_id, RunATenOP, action_param,
              action_input_ids, node_id, std::nullopt,
              0, 0, false, std::nullopt, false);

      // For checkpoint input-----------------------------------------------------------
      for (auto node_input_value: node->inputs()) {
        NodeID node_input_id = node_input_value->unique();
        if (!checkpoint_inputs.contains(node_input_id))
          continue;
        if (this->checkpoint_locations[node_input_id].contains(assigned_to)) {
          this->IssueAction_ToCluster(assigned_to, issuing_id, FetchCheckpoint, std::make_shared<AccActionParamPayload>(),
                  {}, node_input_id, std::nullopt,
                  1, 0, true, Trace({node_input_id, std::nullopt}), false);
        } else {
          Rank checkpoint_src = *(this->checkpoint_locations[node_input_id].cbegin());
          this->IssueAction_ToCluster(checkpoint_src, issuing_id, FetchCheckpoint, std::make_shared<AccActionParamPayload>(),
                  {}, node_input_id, std::nullopt,
                  1, 0, true, Trace({node_input_id, std::nullopt}), false);

          add_transmit(checkpoint_src, assigned_to, node_input_id);

          auto action_param = std::make_shared<struct SettledAsCheckpoint>();
          action_param->node_id = node_input_id;
          this->IssueAction_ToCluster(assigned_to, issuing_id, SettledAsCheckpoint, action_param,
                  {node_input_id}, std::nullopt, Trace({node_input_id, std::nullopt}),
                  1, 0, true, std::nullopt, false);

          this->checkpoint_locations[node_input_id].insert(assigned_to);
        }
      }

    }

    this->ActionCommit();
  }

private:
  std::mutex lock;

  std::unordered_map<Rank, std::shared_ptr<struct WatchdogSignAccIn>> ranks;
  Rank last_rank__sign_in;

  TSGraph GCG;
  std::unordered_map<NodeID, torch::jit::Node *> node_id__to__node;

  std::unordered_map<NodeID, std::unordered_set<Rank>> checkpoint_locations;

  std::unordered_map<TaskID,
      std::pair<TSGraph_ptr, std::unordered_map<std::string, std::string>>> tasks;

  TaskID cur_task_id;
  TransmitID cur_transmit_id;
  std::vector<std::shared_ptr<struct AccRequestSend>> pending_transmits;
  // int has_ongoing_transmit;


  // add for complex transmit
  std::set<Rank> transmit_busy_workers;
  std::unordered_map<std::shared_ptr<struct WorkerPair>, std::set<TransmitID>> transmit_worker_pairs_to_idle_transmit;
  std::unordered_map<std::shared_ptr<struct WorkerPair>, std::set<TransmitID>> transmit_worker_pairs_to_busy_transmit;

  std::unordered_map<TransmitID, std::pair<Rank, std::shared_ptr<struct AccRequestSend>>> transmit_id_to_info;

    //  self.transmit_id_to_info: Dict[TransmitID, Tuple[Rank, Rank, NodeID, RunID, Any]] = dict()

  IssuingID cur_issuing_id;

  RankManager rank_manager_;
};


std::shared_ptr<Master>
GetMiniGCGMaster() {
  return std::make_shared<MiniGCGMaster>();
}
