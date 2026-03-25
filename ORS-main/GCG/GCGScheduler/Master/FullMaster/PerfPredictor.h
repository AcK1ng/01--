#ifndef PERFPREDICTOR_H
#define PERFPREDICTOR_H
#include "Base.h"

class OperatorPerformanceModel {
public:
    virtual void train(Duration time) = 0;
    virtual Duration predict() = 0;
    virtual nlohmann::json ToJson() const = 0;
    friend void to_json(nlohmann::json& j, const OperatorPerformanceModel& t) {
        j = t.ToJson();
    }
};

class TaskPerformancePredictor {
public:
    void new_task(TaskID taskId, TaskNodeID task_node_id_count) {
        std::lock_guard<std::mutex> holding_lock(this->lock);
        for (TaskNodeID task_node_id = 0; task_node_id < task_node_id_count; task_node_id++)
            this->tasks_profiling[{taskId, task_node_id}];
        this->task_node_id_counts.insert({taskId, task_node_id_count});
    }
    
    void drop_task(TaskID taskId) {
        std::lock_guard<std::mutex> holding_lock(this->lock);
        TaskNodeID task_node_id_count = this->task_node_id_counts.at(taskId);
        for (TaskNodeID task_node_id = 0; task_node_id < task_node_id_count; task_node_id++)
            this->tasks_profiling.erase({taskId, task_node_id_count});
        this->task_node_id_counts.erase(taskId);
    }
    
    void train(TaskID taskId, 
               TaskNodeID task_node_id,
               AccModel acc_model,
               Duration time) {
#ifndef NDEBUG
        std::cout << "train operator taskID[" << taskId << "] "
                  << "TaskNodeID[" << task_node_id << "] "
                  << "acc_model[" << acc_model << "] "
                  << "time[" << time << "] "
                  << std::endl;
#endif
        this->__get_model(taskId, task_node_id, acc_model)->train(time);
    }
    
    Duration
    predict(TaskID taskId, TaskNodeID task_node_id, AccModel acc_model) {
        return this->__get_model(taskId, task_node_id, acc_model)->predict();
    }

    virtual nlohmann::json ToJson() const {
        nlohmann::json j;
        j["task_node_id_counts"] = this->task_node_id_counts;
        j["tasks_profiling"] = this->tasks_profiling;
        return j;
    }
    friend void to_json(nlohmann::json& j, const TaskPerformancePredictor& t) {
        j = t.ToJson();
    }
    virtual void FromJson(const nlohmann::json &j) {
        j.at("task_node_id_counts").get_to(this->task_node_id_counts);
        for (auto &all_models__for_op: j.at("tasks_profiling")) {
            auto k = all_models__for_op[0].get<std::tuple<TaskID, TaskNodeID>>();
            this->tasks_profiling[k];
            for (auto &[acc_model, model_json]: all_models__for_op[1].items())
                this->tasks_profiling[k][acc_model] = this->__get_model__from_json(model_json);
        }
    }
private:
    std::mutex lock;
    std::unordered_map<TaskID, TaskNodeID> task_node_id_counts; // TaskID -> Nr_TaskNodeID
    std::unordered_map<std::tuple<TaskID, TaskNodeID>,
                       std::unordered_map<AccModel, std::shared_ptr<OperatorPerformanceModel>>> tasks_profiling;

    inline std::shared_ptr<OperatorPerformanceModel>
    __get_model(TaskID taskId, TaskNodeID task_node_id, AccModel acc_model) {
        std::lock_guard<std::mutex> holding_lock(this->lock);
        assert(this->tasks_profiling.contains({taskId, task_node_id}));
        if (!this->tasks_profiling.at({taskId, task_node_id}).contains(acc_model))
            this->tasks_profiling.at({taskId, task_node_id})[acc_model] = this->__new_model();
        return this->tasks_profiling.at({taskId, task_node_id})[acc_model];
    }

    virtual std::shared_ptr<OperatorPerformanceModel> __new_model() = 0;
    virtual std::shared_ptr<OperatorPerformanceModel>
    __get_model__from_json(const nlohmann::json& j) = 0;
};

extern std::shared_ptr<TaskPerformancePredictor>
GetSimpleTaskPerformancePredictor(int warmup_step = 3, double fix_factor = 0.5);


extern std::shared_ptr<TaskPerformancePredictor>
GetTaskPerformancePredictorFromJson(const nlohmann::json& j);






class TransmitrPerformanceModel {
public:
    virtual void train(NBytes nbytes, Duration time) = 0;
    virtual Duration predict(NBytes nbytes) = 0;
    virtual bool operator<(std::shared_ptr<TransmitrPerformanceModel> other) = 0;
    virtual double bw() = 0;
    virtual nlohmann::json ToJson() const = 0;
    friend void to_json(nlohmann::json& j, const TransmitrPerformanceModel& t) {
        j = t.ToJson();
    }
};

class TransmitPerformancePredictor {
public:
    void sign_in(ComputeDomain new_domain) {
        std::lock_guard<std::mutex> holding_lock(this->lock);
        if (this->domains.contains(new_domain))
            return;
        for (auto old_domain : this->domains) {
            ComputeDomain small_domain = std::min(old_domain, new_domain);
            ComputeDomain big_domain = std::max(old_domain, new_domain);
            this->pair_models[{small_domain, big_domain}] = __new_model();
        }
        this->domains.insert(new_domain);
    }
 
    void sign_out(ComputeDomain old_domain) {
        std::lock_guard<std::mutex> holding_lock(this->lock);
        if (!this->domains.contains(old_domain))
            return;
        for (auto domain: this->domains) {
            Rank small_domain = std::min(old_domain, domain);
            Rank big_domain = std::max(old_domain, domain);
            this->pair_models.erase({small_domain, big_domain});
        }
        this->domains.erase(old_domain);
    }
 
    void train(ComputeDomain one_peer, ComputeDomain another, std::vector<std::pair<NBytes, Duration>> &pairs) {
        auto model = this->__get_model(one_peer, another);
        for(auto [nbytes, time_ns]: pairs)
            model->train(nbytes, time_ns);
    }

    Duration predict(ComputeDomain one_peer, ComputeDomain another, NBytes nbyte) {
        return this->__get_model(one_peer, another)->predict(nbyte);
    }
 
    std::vector<std::pair<ComputeDomain, double>> adjacent_peers(ComputeDomain one_peer) {
        std::vector<std::pair<ComputeDomain, double>> domain_to_bw;
        for (auto another: this->domains) {
            if (another == one_peer) 
                continue;
            domain_to_bw.push_back({another, this->__get_model(one_peer, another)->bw()});
        }
        domain_to_bw.push_back({one_peer, std::numeric_limits<double>::max()});

        // rank_to_bw.push_back(std::make_pair<one_peer, std::numeric_limits<double>::max()>);
        std::sort(domain_to_bw.begin(), domain_to_bw.end(), [](const auto& a, const auto& b) {
            return a.second > b.second; // 降序排序
        });
        return domain_to_bw;
    }

    virtual nlohmann::json ToJson() const {
        nlohmann::json j;
        j["ranks"] = this->domains;
        j["pair__to__model_id"] = {};
        nlohmann::json models_j = nlohmann::json::array();
        size_t model_id = 0;

        std::unordered_map<DomainPair, size_t> pair__to__model_id;
        for (auto &[domain_pair, model]: this->pair_models) {
            pair__to__model_id.insert({domain_pair, model_id});
            models_j.push_back(model);
            model_id++;
        }
        j["pair__to__model_id"] = pair__to__model_id;
        j["models"] = models_j;
        return j;
    }
    friend void to_json(nlohmann::json& j, const TransmitPerformancePredictor& t) {
        j = t.ToJson();
    }
    virtual void FromJson(const nlohmann::json &j) {
        j.at("ranks").get_to(this->domains);
        std::vector<std::shared_ptr<TransmitrPerformanceModel>> models;
        for (auto model: j.at("models"))
            models.push_back(this->__get_model__from_json(model));
            
        std::unordered_map<DomainPair, size_t> pair__to__model_id;
        j.at("pair__to__model_id").get_to(pair__to__model_id);

        for (auto &[domain_pair, model_id]: pair__to__model_id) {
            this->pair_models[domain_pair] = models[model_id];
        }
    }
private:
    std::mutex lock;
    std::unordered_set<ComputeDomain> domains;
    std::unordered_map<DomainPair, std::shared_ptr<TransmitrPerformanceModel>> pair_models;

    inline std::shared_ptr<TransmitrPerformanceModel>
    __get_model(ComputeDomain one_peer, ComputeDomain another) {
        ComputeDomain small_rank = std::min(one_peer, another);
        ComputeDomain big_rank = std::max(one_peer, another);
        std::lock_guard<std::mutex> holding_lock(this->lock);
        return this->pair_models[{small_rank, big_rank}];
    }

    virtual std::shared_ptr<TransmitrPerformanceModel> __new_model() = 0;
    virtual std::shared_ptr<TransmitrPerformanceModel>
    __get_model__from_json(const nlohmann::json& j) = 0;
};

extern std::shared_ptr<TransmitPerformancePredictor>
GetLinearTransmitPerformancePredictor(int warmup_step = 3, double lr = 0.01);

extern std::shared_ptr<TransmitPerformancePredictor>
GetTransmitPerformancePredictorFromJson(const nlohmann::json& j);


#endif