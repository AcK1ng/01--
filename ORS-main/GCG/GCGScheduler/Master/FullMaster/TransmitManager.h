#ifndef TRANSMIT_MANAGER_H
#define TRANSMIT_MANAGER_H

#include "FullMasterBase.h"

#include <map>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <mutex>
#include <algorithm>
#include <list>
#include <queue>
#include <vector>
#include <functional>
#include <set>
#include <random>

class TransmitManager {
public:
    AliveIDs alive_tranmit_ids;
    std::unordered_map<NodeID, std::unordered_set<TransmitID>> node_id__to__transmit_ids;
    std::unordered_map<std::tuple<NodeID, Rank>, TransmitID> recv__to__transmit_id; // 主要照顾到模拟器有得出node_id何时到达recv的时间戳
    std::unordered_map<TransmitID, std::shared_ptr<struct TransmitInfo>> transmit_id__to__info;

protected:
    std::function<Timestamp ()> Now;

    std::vector<
        std::function<void (std::shared_ptr<struct TransmitInfo>)>> callbacks__for_transmit_permitted;
    std::vector<
        std::function<void (std::shared_ptr<struct TransmitInfo>)>> callbacks__for_new_transmit;

private:
    void
    shared_init() {
        auto permit_callback = [this](std::shared_ptr<struct TransmitInfo> transmit_info) {
            transmit_info->has_been_permitted = true;
            transmit_info->permit_time = this->Now();
        };
        this->__register_callback_for_transmit_permit(permit_callback);
    }

public:
    TransmitManager() {
        this->shared_init();
    }

    TransmitManager(const TransmitManager &another) {
        this->alive_tranmit_ids = another.alive_tranmit_ids;
        this->node_id__to__transmit_ids = another.node_id__to__transmit_ids;
        this->recv__to__transmit_id = another.recv__to__transmit_id;
        this->transmit_id__to__info = another.transmit_id__to__info;
        for (auto &p: another.transmit_id__to__info) {
            TransmitID transmit_id = p.first;
            auto transmit_info = p.second;
            this->transmit_id__to__info[transmit_id] = std::make_shared<TransmitInfo>(*(transmit_info.get()));
        }
        this->shared_init();
    }

    virtual std::unique_ptr<TransmitManager>
    _clone() const {
        return std::make_unique<TransmitManager>(*this);
    }

    virtual nlohmann::json
    ToJson() const {
        nlohmann::json j;
        j["alive_tranmit_ids"] = this->alive_tranmit_ids;
        j["node_id__to__transmit_ids"] = this->node_id__to__transmit_ids;
        j["recv__to__transmit_id"] = this->recv__to__transmit_id;
        j["transmit_id__to__info"] = this->transmit_id__to__info;
        return j;
    }

    friend void
    to_json(nlohmann::json& j, const TransmitManager& t) {
        j = t.ToJson();
    }

    virtual void
    FromJson(const nlohmann::json &j) {
        j.at("alive_tranmit_ids").get_to(this->alive_tranmit_ids);
        j.at("node_id__to__transmit_ids").get_to(this->node_id__to__transmit_ids);
        j.at("recv__to__transmit_id").get_to(this->recv__to__transmit_id);
        j.at("transmit_id__to__info").get_to(this->transmit_id__to__info);
        this->shared_init();
    }

    void
    __register_timer(std::function<Timestamp ()> timer) {
        this->Now = timer;
    }

    void
    __register_callback_for_transmit_permit(
        std::function<void (std::shared_ptr<struct TransmitInfo>)> callback) {
        this->callbacks__for_transmit_permitted.push_back(callback);
    }

    void
    __register_callback_for_new_transmit(
        std::function<void (std::shared_ptr<struct TransmitInfo>)> callback) {
        this->callbacks__for_new_transmit.push_back(callback);
    }

    inline std::shared_ptr<struct TransmitInfo>
    GetTransmitInfo(TransmitID transmit_id) {
        return this->transmit_id__to__info[transmit_id];
    }

    std::shared_ptr<struct TransmitInfo>
    NewTransmitPair(Rank send, std::string send_resource, ComputeDomain send_domain,
                    Rank recv, std::string recv_resource, ComputeDomain recv_domain,
                    NodeID node_id,
                    bool recv_for_settled_ckpt,
                    std::optional<IssuingID> issuing_id = std::nullopt) {
        auto transmit_info = std::make_shared<struct TransmitInfo>();

        TransmitID transmit_id1 = this->alive_tranmit_ids.allocateID();
        TransmitID transmit_id2 = this->alive_tranmit_ids.allocateID();

        this->transmit_id__to__info[transmit_id1] = transmit_info;

        transmit_info->transmit_id = transmit_id1;
        transmit_info->transmit_id2 = transmit_id2;
        transmit_info->node__to_transmit = node_id;
        transmit_info->send_resource = send_resource;
        transmit_info->send_rank = send;
        transmit_info->send_host = 0;
        transmit_info->send_domain = send_domain;
        transmit_info->recv_resource = recv_resource;
        transmit_info->recv_rank = recv;
        transmit_info->recv_host = 0;
        transmit_info->recv_domain = recv_domain;
        transmit_info->recv_for_settled_ckpt = recv_for_settled_ckpt;
        transmit_info->has_requested_send = false;
        transmit_info->has_been_permitted = false;
        transmit_info->has_finished = false;
        transmit_info->issuing_id = issuing_id;

        if (!this->node_id__to__transmit_ids.contains(node_id))
            this->node_id__to__transmit_ids.insert({});
        this->node_id__to__transmit_ids[node_id].insert(transmit_id1);
        this->recv__to__transmit_id[{node_id, recv}] = transmit_id1;

        for (auto &callback: this->callbacks__for_new_transmit)
            callback(transmit_info);

        return transmit_info;
    }

    // 这里是信息正常删除
    void
    PurgeInfoAboutNode(NodeID node_id) {
        if (!this->node_id__to__transmit_ids.contains(node_id))
            return;
        for (auto transmit_id: this->node_id__to__transmit_ids[node_id]) {
            assert(transmit_id__to__info[transmit_id]->has_finished);
            auto transmit_info = this->transmit_id__to__info[transmit_id];
            this->recv__to__transmit_id.erase({node_id, transmit_info->recv_rank});
            this->transmit_id__to__info.erase(transmit_id);
            this->alive_tranmit_ids.releaseID(transmit_id);
        }
        this->node_id__to__transmit_ids.erase(node_id);
    }

    // 这里是容错情况，删掉和这个issuing_id有关的所有东西
    virtual void
    WithdrawIssuingID(IssuingID issuing_id) {
        std::unordered_set<TransmitID> transmit_to_be_deleted;
        for (auto &p: this->transmit_id__to__info) {
            TransmitID transmit_id = p.first;
            auto transmit_info = p.second;
            if (transmit_info->issuing_id != issuing_id)
                continue;

            transmit_to_be_deleted.insert(transmit_id);
        }

        for (auto transmit_id: transmit_to_be_deleted) {
            auto transmit_info = this->transmit_id__to__info[transmit_id];
            NodeID node_id = transmit_info->node__to_transmit;
            this->node_id__to__transmit_ids[node_id].erase(transmit_id);
            if (this->node_id__to__transmit_ids[node_id].empty())
                this->node_id__to__transmit_ids.erase(node_id);
            this->transmit_id__to__info.erase(transmit_id);
            this->alive_tranmit_ids.releaseID(transmit_id);
            this->recv__to__transmit_id.erase({transmit_info->node__to_transmit, transmit_info->recv_rank});
        }
    }

    virtual bool
    RequestTransmitStart(TransmitID transmit_id,
                         std::shared_ptr<VariableDescriptor> variable_descriptor) {
        auto transmit_info = this->transmit_id__to__info[transmit_id];
        if (transmit_info->has_requested_send == true)
            return false;
        transmit_info->has_requested_send = true;
        transmit_info->variable_descriptor = variable_descriptor;
        return true;
    }

    virtual bool
    ReportTransmitDone(TransmitID transmit_id, std::shared_ptr<struct AccReportRecvDone> param) {
        auto transmit_info = this->transmit_id__to__info[transmit_id];
        if (transmit_info->has_finished == true)
            return false;
        transmit_info->has_finished = true;
        transmit_info->finish_time = this->Now();
        return true;
    }
};

class SimpleTransmitManager: public TransmitManager {
private:
    std::unordered_set<Rank> transmit__busy_workers;
    std::unordered_map<RankPair, std::unordered_set<TransmitID>> transmit__ready_and_unpermitted;
    std::unordered_map<RankPair, std::unordered_set<TransmitID>> transmit__permitted_transmit;

public:
    SimpleTransmitManager() {
    }

    SimpleTransmitManager(const SimpleTransmitManager &another): TransmitManager(another) {
        this->transmit__busy_workers = another.transmit__busy_workers;
        this->transmit__ready_and_unpermitted = another.transmit__ready_and_unpermitted;
        this->transmit__permitted_transmit = another.transmit__permitted_transmit;
    }

    virtual std::unique_ptr<TransmitManager>
    _clone() const override {
        return std::make_unique<SimpleTransmitManager>(*this);
    }

    virtual nlohmann::json
    ToJson() const override {
        nlohmann::json j;
        j["type"] = "SimpleTransmitManager";
        j["transmit__busy_workers"] = this->transmit__busy_workers;
        j["transmit__ready_and_unpermitted"] = this->transmit__ready_and_unpermitted;
        j["transmit__permitted_transmit"] = this->transmit__permitted_transmit;
        nlohmann::json parent = this->TransmitManager::ToJson();
        j.insert(parent.begin(), parent.end());
        return j;
    }

    virtual void
    FromJson(const nlohmann::json &j) override {
        this->TransmitManager::FromJson(j);
        j.at("transmit__busy_workers").get_to(this->transmit__busy_workers);
        j.at("transmit__ready_and_unpermitted").get_to(this->transmit__ready_and_unpermitted);
        j.at("transmit__permitted_transmit").get_to(this->transmit__permitted_transmit);
    }

    virtual void
    WithdrawIssuingID(IssuingID issuing_id) {
        this->TransmitManager::WithdrawIssuingID(issuing_id);
        assert(0);
    }

    virtual bool
    RequestTransmitStart(TransmitID transmit_id,
                         std::shared_ptr<VariableDescriptor> variable_descriptor) {
        bool ret = this->TransmitManager::RequestTransmitStart(transmit_id, variable_descriptor);
        if (ret == false)
            return false;

        auto transmit_info = this->transmit_id__to__info[transmit_id];
        Rank sender = transmit_info->send_rank;
        Rank receiver = transmit_info->recv_rank;

        if (this->__can_transmit_start(sender, receiver)) {
            for (auto &callback: this->callbacks__for_transmit_permitted)
                callback(transmit_info);

            if (!this->transmit__permitted_transmit.contains({sender, receiver})) {
                this->transmit__busy_workers.insert(sender);
                this->transmit__busy_workers.insert(receiver);
                this->transmit__permitted_transmit[{sender, receiver}] = std::unordered_set<TransmitID>();
            }
            this->transmit__permitted_transmit[{sender, receiver}].insert(transmit_id);
        } else {
            if (!this->transmit__ready_and_unpermitted.contains({sender, receiver}))
                this->transmit__ready_and_unpermitted[{sender, receiver}] = std::unordered_set<TransmitID>();
            this->transmit__ready_and_unpermitted[{sender, receiver}].insert(transmit_info->transmit_id);
        }

        return true;
    }

    virtual bool
    ReportTransmitDone(TransmitID transmit_id, std::shared_ptr<struct AccReportRecvDone> param) {
        bool ret = this->TransmitManager::ReportTransmitDone(transmit_id, param);
        if (ret == false)
            return false;

        auto transmit_info = this->transmit_id__to__info[transmit_id];

        Rank sender = transmit_info->send_rank;
        Rank receiver = transmit_info->recv_rank;

        this->transmit__permitted_transmit[{sender, receiver}].erase(transmit_info->transmit_id);

        if (this->transmit__permitted_transmit[{sender, receiver}].size() == 0) {
            this->transmit__permitted_transmit.erase({sender, receiver});
            this->transmit__busy_workers.erase(sender);
            this->transmit__busy_workers.erase(receiver);
        }

        auto started_idle_worker_pair = std::unordered_set<RankPair>();

        for (auto &entry : this->transmit__ready_and_unpermitted) {
            auto &idle_worker_pair = entry.first;
            Rank sender, receiver;
            std::tie(sender, receiver) = idle_worker_pair;
            if (__can_transmit_start(sender, receiver)) {
                this->transmit__permitted_transmit[idle_worker_pair] = entry.second;
                this->transmit__busy_workers.insert(sender);
                this->transmit__busy_workers.insert(receiver);

                for (TransmitID idle_transmit_id: entry.second) {
                    auto idle_transmit_info = this->transmit_id__to__info[idle_transmit_id];

                    for (auto &callback: this->callbacks__for_transmit_permitted)
                        callback(idle_transmit_info);
                }
                started_idle_worker_pair.insert({sender, receiver});
            }
        }

        for (auto &worker_pair: started_idle_worker_pair) {
            Rank sender, receiver;
            std::tie(sender, receiver) = worker_pair;
            this->transmit__ready_and_unpermitted.erase({sender, receiver});
        }

        return true;
    }

private:
    inline bool
    __can_transmit_start(Rank sender, Rank receiver) {
        if (// 要么这个worker_pair已经permitted
            this->transmit__permitted_transmit.contains({sender, receiver})

            || 

            // 要么两个worker都不忙
            (!this->transmit__busy_workers.contains(sender)
             && !this->transmit__busy_workers.contains(receiver)))
            return true;
        return false;
    };
};

extern std::shared_ptr<SimpleTransmitManager>
Get_SimpleTransmitManager();

extern std::shared_ptr<SimpleTransmitManager>
Get_TransmitManager_FromJson(const nlohmann::json &);

#endif