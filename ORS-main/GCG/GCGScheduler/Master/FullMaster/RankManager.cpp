#include "RankManager.h"
#include <fstream>
#include <iostream>

using json = nlohmann::json;

RankManager::RankManager() 
  : transmit_predictor_(GetLinearTransmitPerformancePredictor()) {}                                     

void RankManager::sign_in(std::shared_ptr<struct WatchdogSignAccIn> sign_in_info) {
    if (!this->domain__to__capacity.contains(sign_in_info->domain))
        this->domain__to__capacity[sign_in_info->domain] = sign_in_info->hbm_capability;
    this->sign_in(sign_in_info->rank, sign_in_info->host_id, sign_in_info->acc_model, sign_in_info->resource, sign_in_info->domain);
    nodes_[sign_in_info->rank].sign_in_info = std::make_shared<struct WatchdogSignAccIn>(*(sign_in_info.get()));
}

void RankManager::sign_in(Rank rank, HostID host_id, const std::string& model, const std::string& resource, ComputeDomain domain) {
    // 存储节点信息
    nodes_[rank] = {rank, host_id, model, resource};

    this->rank__to__domain[rank] = domain;
    if (!this->domain__to__ranks.contains(domain))
        this->domain__to__ranks[domain];
    this->domain__to__ranks.at(domain).insert(rank);
    
    // 初始化节点图结构
    graph_[rank] = {};
    
    // 更新host_id分组
    hosts_[host_id].insert(rank);
    host_models_[host_id] = model;
    host_resources_[host_id] = resource;

    if (!this->models_set.contains(model)) {
        this->models_set.insert(model);
        this->models.push_back(model);
    }

    transmit_predictor_->sign_in(domain);
}

void RankManager::sign_out(Rank rank) {
    if (nodes_.find(rank) == nodes_.end()) return;
    
    const auto& node_info = nodes_.at(rank);
    HostID host_id = node_info.host_id;
    ComputeDomain domain = this->rank__to__domain.at(rank);
    
    nodes_.erase(rank);

    this->rank__to__domain.erase(rank);
    this->domain__to__ranks.at(domain).erase(rank);

    auto host_it = hosts_.find(host_id);
    if (host_it != hosts_.end()) {
        host_it->second.erase(rank);
        if (host_it->second.empty()) {
            hosts_.erase(host_it);
            host_models_.erase(host_id);
            host_resources_.erase(host_id);
        }
    }
    
    graph_.erase(rank);
    for (auto& [_, neighbors] : graph_) {
        neighbors.erase(rank);
    }

    if (this->domain__to__ranks.at(domain).empty()) {
        this->domain__to__capacity.erase(domain);
        transmit_predictor_->sign_out(domain);
    }
}

bool RankManager::IsReliable(Rank rank) {
    return this->nodes_[rank].sign_in_info->never_signout;
}

std::vector<Rank> RankManager::GetTopKNeighbors(Rank node, int k) {
    if (graph_.find(node) == graph_.end()) return {};
    
    using SpeedRank = std::pair<double, Rank>;
    std::priority_queue<SpeedRank, std::vector<SpeedRank>, std::greater<SpeedRank>> min_heap;
    
    for (const auto& [neighbor, speed] : graph_[node]) {
        if (min_heap.size() < static_cast<size_t>(k)) {
            min_heap.push({speed, neighbor});
        } else if (speed > min_heap.top().first) {
            min_heap.pop();
            min_heap.push({speed, neighbor});
        }
    }
    
    std::vector<Rank> result;
    while (!min_heap.empty()) {
        result.push_back(min_heap.top().second);
        min_heap.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

const std::set<Rank>& RankManager::GetRanksByHost(HostID host_id) const {
    static const std::set<Rank> empty_set;
    auto it = hosts_.find(host_id);
    return (it != hosts_.end()) ? it->second : empty_set;
}

std::string RankManager::GetModelByHost(HostID host_id) const {
    auto it = host_models_.find(host_id);
    return (it != host_models_.end()) ? it->second : "";
}

std::string RankManager::GetStandardModelByTaskNodeID(TaskNodeID task_node_id) const {
    return this->models[task_node_id % this->models.size()];
}

std::string RankManager::GetResourceByHost(HostID host_id) const {
    auto it = host_resources_.find(host_id);
    return (it != host_resources_.end()) ? it->second : "";
}

HostID RankManager::GetHostByRank(Rank rank) const {
    auto it = nodes_.find(rank);
    return (it != nodes_.end()) ? it->second.host_id : -1;
}

size_t RankManager::GetHBMCapability(Rank rank) const {
    auto it = nodes_.find(rank);
    return (it != nodes_.end()) ? it->second.sign_in_info->hbm_capability : -1;
}

std::vector<HostID> RankManager::GetAllHosts() const {
    std::vector<HostID> hosts;
    for (const auto& [host_id, _] : hosts_) {
        hosts.push_back(host_id);
    }
    return hosts;
}

std::vector<Rank> RankManager::GetAllRanks() const {
    std::vector<Rank> ranks;
    for (const auto& [rank, _] : nodes_) {
        ranks.push_back(rank);
    }
    return ranks;
}

const std::unordered_map<ComputeDomain, NBytes> &RankManager::GetAllComputeDomains() const {
    return this->domain__to__capacity;
}

void RankManager::PrintTopologySummary() {
    std::cout << "\n====== Node Topology Summary ======\n";
    std::cout << "Total Hosts: " << hosts_.size() << "\n";
    std::cout << "Total Ranks: " << nodes_.size() << "\n\n";
    
    for (const auto& [host_id, ranks] : hosts_) {
        std::cout << "Host[" << host_id << "] Model[" << host_models_.at(host_id)
                  << "] Resource[" << host_resources_.at(host_id) << "]\n";
        std::cout << "  Ranks: ";
        for (Rank rank : ranks) {
            std::cout << rank << " ";
        }
        std::cout << "\n";
        
        for (Rank rank : ranks) {
            if (graph_.find(rank) != graph_.end()) {
                auto fastest = GetTopKNeighbors(rank, 1);
                if (!fastest.empty()) {
                    std::cout << "    Rank[" << rank << "] Fastest neighbor: Rank[" 
                              << fastest[0] << "]\n";
                }
            }
        }
    }
    std::cout << "==================================\n";
}

void RankManager::TrainTransmit(Rank src, Rank dst, NBytes nbytes, Duration time) {
    ComputeDomain src_domain = this->rank__to__domain.at(src);
    ComputeDomain dst_domain = this->rank__to__domain.at(dst);
    if (src_domain == dst_domain)
        return;
    std::vector<std::pair<NBytes, Duration>> pairs = {{nbytes, time}};
    transmit_predictor_->train(src_domain, dst_domain, pairs);
}

Duration RankManager::PredictTransmit(Rank src, Rank dst, NBytes nbyte) {
    ComputeDomain src_domain = this->rank__to__domain.at(src);
    ComputeDomain dst_domain = this->rank__to__domain.at(dst);
    if (src_domain == dst_domain)
        return 0;
    auto prediction = transmit_predictor_->predict(src_domain, dst_domain, nbyte);
    return prediction;
}

double RankManager::GetBW(Rank src, Rank dst) {
    ComputeDomain src_domain = this->rank__to__domain.at(src);
    ComputeDomain dst_domain = this->rank__to__domain.at(dst);
    auto peers = transmit_predictor_->adjacent_peers(src_domain);
    for (const auto& [peer, bw] : peers) {
        if (peer == dst_domain) return bw;
    }
    return -1.0; // 返回负值表示未找到
}

Rank RankManager::GetRandomOnlineRank() const {
    if (nodes_.empty()) return -1;
    
    auto it = nodes_.begin();
    std::advance(it, std::rand() % nodes_.size());
    return it->first;
}

ComputeDomain RankManager::GetComputeDomain(Rank rank) const {
    return this->rank__to__domain.at(rank);
}

nlohmann::json RankManager::ToJson() const {
    json j;
    j["nodes"] = nodes_;
    j["rank__to__domain"] = rank__to__domain;
    j["domain__to__ranks"] = domain__to__ranks;
    j["domain__to__capacity"] = domain__to__capacity;
    j["graph"] = graph_;
    j["hosts"] = hosts_;
    j["host_models"] = host_models_;
    j["host_resources"] = host_resources_;
    j["transmit_predictor"] = transmit_predictor_;
    j["models_set"] = models_set;
    j["models"] = models;
    return j;
}


void RankManager::FromJson(const nlohmann::json& j) {
    j.at("nodes").get_to(nodes_);
    j.at("rank__to__domain").get_to(rank__to__domain);
    j.at("domain__to__ranks").get_to(domain__to__ranks);
    j.at("domain__to__capacity").get_to(domain__to__capacity);
    j.at("graph").get_to(graph_);
    j.at("hosts").get_to(hosts_);
    j.at("host_models").get_to(host_models_);
    j.at("host_resources").get_to(host_resources_);
    j.at("models_set").get_to(models_set);
    j.at("models").get_to(models);
    this->transmit_predictor_ = GetTransmitPerformancePredictorFromJson(j.at("transmit_predictor"));
}


// 保存到文件
void RankManager::SaveToFile(const std::string& filename) const {
    std::ofstream file(filename); 
    file << ToJson().dump(4); 
}

// 从文件加载
void RankManager::LoadFromFile(const std::string& filename) {
    std::ifstream file(filename); 
    json j;
    file >> j;
    FromJson(j);
}