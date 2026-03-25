#include "Base.h"
#include "Master/Master.h"

#include <unordered_map>

std::ostream&
operator<<(std::ostream &os, const Trace &trace) {
  os << "Trace[";

  auto node_id = std::get<0>(trace);
  os << "NodeID[";
  if (node_id.has_value())
    os << node_id.value();
  else
    os << "null";
  os << "]";

  os << ",";

  auto transmit_id = std::get<1>(trace);
  os << "TransmitID[";
  if (transmit_id.has_value())
    os << transmit_id.value();
  else
    os << "null";
  os << "]";

  os << "]";
  return os;
}

static std::unordered_map<AccActionEnum, std::string>
acc_action__to__str = {
#define DEF_ACC_ACTION_STR(e) {e, #e},
FOR_EACH_ACC_ACTION(DEF_ACC_ACTION_STR)
#undef DEF_ACC_ACTION_STR
};

std::ostream&
operator<<(std::ostream &os, const struct AccActionBase &action_base) {
  nlohmann::json j = action_base;
  os << j.dump();
  return os;
}

std::ostream&
operator<<(std::ostream &os, const struct AccActionSpec &action_spec) {
  nlohmann::json j = action_spec;
  os << j.dump();
  return os;
}

std::ostream&
operator<<(std::ostream &os, const struct TransmitInfo &transmit_info) {
  // nlohmann::json j = transmit_info;
  //os << j.dump();

  os << "NodeID[" << transmit_info.node__to_transmit << "]: "
     << "Rank[" << transmit_info.send_rank << "] "
     << "-> Rank[" << transmit_info.recv_rank << "] "
     << "(TransmitID[" << transmit_info.transmit_id << "]) "
     << "has_requested_send[" << transmit_info.has_requested_send << "] "
     << "has_been_permitted[" << transmit_info.has_been_permitted << "] "
     << "has_finished[" << transmit_info.has_finished << "] ";
  return os;
}
void
Master::FreeCheckpoint(Rank rank, NodeID node_id) {
  this->cluster->FreeCheckpoint(rank, node_id);
}
void
Master::IssueActions_ToCluster(Rank rank,
                               std::optional<IssuingID> issuing_id,
                               std::vector<std::shared_ptr<AccActionSpec>> &action_specs) {
  this->cluster->IssueActions(rank, issuing_id, action_specs);
}
void
Master::PermitRecvs_InCluster(Rank rank,
                              std::vector<std::shared_ptr<TransmitSpec>> &transmit_specs) {
  this->cluster->PermitRecvs(rank, transmit_specs);
}
void
Master::WithdrawActions_FromCluster(Rank rank, IssuingID issuing_id) {
  this->cluster->WithdrawActions(rank, issuing_id);
}



std::shared_ptr<Master>
Master::GetMasterFromJson(const nlohmann::json &j) {

  std::string type = j.at("type").get<std::string>();
  if (type == "SimpleFullMaster") {
    auto _m = GetSimpleFullMasterFromJson(j);
    return _m;
  } else
    assert(0);
  return nullptr;
}