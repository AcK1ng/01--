#ifndef RANK_MANAGER_H
#define RANK_MANAGER_H

#include "Base.h"
#include "PerfPredictor.h"
#include <map>
#include <unordered_map>
#include <set>
#include <vector>
#include <queue>
#include <mutex>
#include <iostream>
#include <algorithm>

class RankManager {
public:
    struct NodeInfo {
        Rank rank;
        HostID host_id;
        std::string model;
        std::string resource;
        std::shared_ptr<struct WatchdogSignAccIn> sign_in_info;

        NLOHMANN_DEFINE_TYPE_INTRUSIVE(NodeInfo, rank, host_id, model, resource, sign_in_info) //JSON序列化支持
    };

    RankManager();
    void sign_in(std::shared_ptr<struct WatchdogSignAccIn> sign_in_info);
    void sign_in(Rank rank, HostID host_id, const std::string& model, const std::string& resource, ComputeDomain);
    void sign_out(Rank rank);
    bool IsReliable(Rank rank);
    std::vector<Rank> GetTopKNeighbors(Rank node, int k);
    const std::set<Rank>& GetRanksByHost(HostID host_id) const;
    std::string GetModelByHost(HostID host_id) const;
    std::string GetResourceByHost(HostID host_id) const;
    HostID GetHostByRank(Rank rank) const;
    size_t GetHBMCapability(Rank rank) const;
    std::vector<HostID> GetAllHosts() const;
    std::vector<Rank> GetAllRanks() const;
    const std::unordered_map<ComputeDomain, NBytes> &GetAllComputeDomains() const;
    void PrintTopologySummary();
    Rank GetRandomOnlineRank() const;
    ComputeDomain GetComputeDomain(Rank rank) const;

    // 设计这个接口，是为了给定task_node_id，返回一个加速器型号，作为归一化的标准型号
    std::string GetStandardModelByTaskNodeID(TaskNodeID task_node_id) const;

     // 传输性能训练接口
    void TrainTransmit(Rank src, Rank dst, NBytes nbytes, Duration time);
    // 传输性能预测接口
    Duration PredictTransmit(Rank src, Rank dst, NBytes nbytes);
    // 获取节点间带宽
    double GetBW(Rank src, Rank dst);

    friend void
    to_json(nlohmann::json& j, const RankManager &t) {
        j = t.ToJson();
    }
        
    friend void
    from_json(const nlohmann::json& j, RankManager &t) {
        t.FromJson(j);
    }

    nlohmann::json ToJson() const;
    void FromJson(const nlohmann::json& json);
    void SaveToFile(const std::string& filename) const;
    void LoadFromFile(const std::string& filename);

private:
    std::map<Rank, NodeInfo> nodes_;

    std::unordered_map<Rank, ComputeDomain> rank__to__domain;
    std::unordered_map<ComputeDomain, std::unordered_set<Rank>> domain__to__ranks;
    std::unordered_map<ComputeDomain, NBytes> domain__to__capacity;

    std::unordered_map<Rank, std::unordered_map<Rank, double>> graph_;
    std::map<HostID, std::set<Rank>> hosts_;
    std::map<HostID, std::string> host_models_;
    std::map<HostID, std::string> host_resources_;

    // 这两个成员变量，首先为了存加速器型号全集；而后为了给定int作为哈希值，直接返回一个加速器
    std::unordered_set<std::string> models_set;
    std::vector<std::string> models;

    std::shared_ptr<TransmitPerformancePredictor> transmit_predictor_;
};

#endif // RANK_MANAGER_H