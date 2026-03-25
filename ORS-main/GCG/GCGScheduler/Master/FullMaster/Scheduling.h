#ifndef DYNAMIC_SCHEDULER_H
#define DYNAMIC_SCHEDULER_H

#include "Base.h"
#include "Master/FullMaster/FullMasterBase.h"
#include "Master/FullMaster/GCG.h"
#include "Master/FullMaster/TransmitManager.h"
#include "Master/FullMaster/RankManager.h"

#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <algorithm>
#include <limits>
#include <memory>
#include <json.hpp>

#define OP_STREAM 0


class UtilizationStatus {
public:
    UtilizationStatus() = default;

    UtilizationStatus(std::shared_ptr<Simulator> simulator):
        rank__to__busy_duration(simulator->GetMaxRank(), 0),
        rank__to__available_time(simulator->GetMaxRank(), simulator->Now()),
        domain__to__hbm_usage(simulator->GetMaxComputeDomain(), 0) {


        auto action_done_callback =
            [&](Rank r, std::shared_ptr<const AccActionSpec> action,
                Timestamp end, Duration duration) {
                if (action->op_enum == RunATenOP || action->op_enum == UploadTensor)
                    this->scheduled_op_end_time->insert({action->output_id.value(), end});

                this->rank__to__busy_duration[r] += duration;
                this->rank__to__available_time[r] = end;

        };

        simulator->__register_action_done_callback(action_done_callback);

        this->simulation_start_simulation = simulator->Now();

        this->scheduled_op_end_time = std::make_shared<std::unordered_map<NodeID, Timestamp>>();

        while (simulator->Step() != EmptyAction)
            ;

        this->simulation_end_simulation = simulator->Now();


        for (ComputeDomain domain = 0; domain < simulator->GetMaxComputeDomain(); domain++) {
            this->domain__to__hbm_usage[domain] = simulator->GetHBMUsage(domain);

#ifdef UTILIZATION_DEBUG
            std::cout << "ComputeDomain[" << domain << "] " << " HBM Used " << this->domain__to__hbm_usage[domain] << std::endl;
#endif
        }

        this->last_op_end_timestamp = 0;
    }

    UtilizationStatus(const UtilizationStatus &u) = default;

    // transmit/compute 这两个函数设计逻辑是这样
    // 首先假设计算图的执行按照严格拓扑序，就没有计算图内并行
    // 所以说定义了last_op_end_timestamp，来标志上一个算子的结束时间；
    // 或者换句话说，下一个算子的开始一定要在last_op_end_timestamp之后
    //
    // 每个加速器上的available_time指的是，这个加速器可以开始执行算子的时间
    // 在这个模型中我们就先不搞算子插空了，太复杂。算子的执行只能往后排，所以available_time升序的
    //
    // 对于transmit，我们希望察觉搬运超大tensor的代价，就是搬运模型权重
    // 所以大部分transmit时间无足轻重。
    // 当transmit中间变量时，transmit的开始时间为max of (last_op_end_timestamp, available_time)
    // 而搬运超大tensor能够不受last_op_end_timestamp约束，只和avaliable_time有关
    // 所以定义了transmit_for_scheduling_node为False时，来绕过last_op_end_timestamp
    //
    // 当transmit模型权重时，transmit_for_scheduling_node应该为False
    // 当transmit中间变量时，transmit_for_scheduling_node应该为True
    //
    // 因为一个算子的开始时间一定要在上个算子的compute，和所有input的transmit之后，
    // 所以compute和transmit之后要更新last_op_end_timestamp

    Timestamp
    transmit(Rank s, Rank r,
             ComputeDomain domain,
             Duration duration,
             NBytes transmit_size,
             bool transmit_for_scheduling_node) {
        Timestamp start_time = std::max(this->rank__to__available_time[s], this->rank__to__available_time[r]);
        if (transmit_for_scheduling_node == true)
            start_time = std::max(start_time, this->last_op_end_timestamp);
        Timestamp end_time = start_time + duration;

        this->rank__to__available_time[s] = end_time;
        this->rank__to__available_time[r] = end_time;

        this->rank__to__busy_duration[s] += duration;
        this->rank__to__busy_duration[r] += duration;

        this->domain__to__hbm_usage[domain] += transmit_size;

        this->update_end_timestamp(end_time);

        return end_time;
    }

    Timestamp
    compute(Rank r, ComputeDomain domain, Duration duration, NBytes output) {
        Timestamp start_time = std::max(this->rank__to__available_time[r], this->last_op_end_timestamp);
        Timestamp end_time = start_time + duration;

        this->rank__to__available_time[r] = end_time;

        this->rank__to__busy_duration[r] += duration;

        this->domain__to__hbm_usage[domain] += output;

        this->update_end_timestamp(end_time);

        return end_time;
    }

    double
    CalculateUtilizationVariance() {
        Timestamp end = this->last_op_end_timestamp == 0 ? this->simulation_end_simulation : this->last_op_end_timestamp;
        Duration total_duration = end - this->simulation_start_simulation;

        size_t nr_ranks = this->rank__to__busy_duration.size();

        if (total_duration == 0)
            return 0;

        double E_U = 0;
        double E_U_2 = 0;
#ifdef UTILIZATION_DEBUG
        std::cout << "total_duration: " << total_duration << std::endl;
#endif
        for (Duration d: this->rank__to__busy_duration) {
            double utilization;
            if (total_duration == 0)
                utilization = 0;
            else
                utilization = (double)d / total_duration;

#ifdef UTILIZATION_DEBUG
            std::cout << "d: " << d << " Utilization " << utilization << std::endl;
#endif

            E_U += utilization / nr_ranks;
            E_U_2 += (utilization * utilization) / nr_ranks;
        }

        return E_U_2 - E_U * E_U;
    }

    std::shared_ptr<std::unordered_map<NodeID, Timestamp>> scheduled_op_end_time;

    std::vector<Timestamp> rank__to__available_time;
    std::vector<NBytes> domain__to__hbm_usage;

private:

    // last_op_end_timestamp在这定义为，能让下一个算子开始的最早时间
    void
    update_end_timestamp(Timestamp end_timestamp) {
        if (this->last_op_end_timestamp < end_timestamp)
            this->last_op_end_timestamp = end_timestamp;
    }

    std::vector<Duration> rank__to__busy_duration;
    Timestamp last_op_end_timestamp;
    Timestamp simulation_start_simulation, simulation_end_simulation;
};


struct ScoreBreakdown {
    double raw_comp_ratio = 0.0;
    double raw_comm_ratio = 0.0;
    double raw_wait_ratio = 0.0;
    double raw_mem_val    = 0.0;
    double raw_bal_val    = 0.0;

    friend void
    to_json(nlohmann::json& j, const ScoreBreakdown &s) {
        j["raw_comp_ratio"] = s.raw_comp_ratio;
        j["raw_comm_ratio"] = s.raw_comm_ratio;
        j["raw_wait_ratio"] = s.raw_wait_ratio;
        j["raw_mem_val"] = s.raw_mem_val;
        j["raw_bal_val"] = s.raw_bal_val;
    }
};

struct SearchBranch {
    struct UtilizationStatus utilization_status;
    std::unordered_map<NodeID, Rank> placement;
    std::unordered_map<NodeID, Timestamp> end_timestamps;
    size_t node_idx__to_be_scheduled = 0;
    double cost_so_far = 0.0;
    double LB = 0.0;
#ifdef TEST_SCHEDULING_ALGORITHM
    std::vector<ScoreBreakdown> debug_breakdowns;
#endif

    friend void
    to_json(nlohmann::json& j, const SearchBranch &sb) {
        j["placement"] = sb.placement;
        j["cost_so_far"] = sb.cost_so_far;
#ifdef TEST_SCHEDULING_ALGORITHM
        j["debug_breakdowns"] = sb.debug_breakdowns;
#endif
    }
};

class DynamicScheduler {
public:
    // 构造函数
    DynamicScheduler(
        std::function<Duration (AccModel, TaskID, TaskNodeID)> task_predictor,
        std::function<std::pair<Duration, NBytes> (Rank, Rank, std::shared_ptr<VariableDescriptor>)> transmit_predictor,
        std::shared_ptr<GCG_Adding_OpManagement> GCG,
        std::shared_ptr<RankManager> rank_manager,
        std::function<std::shared_ptr<Simulator> (void)> get_simulator
    ): task_predictor(task_predictor),
       transmit_predictor(transmit_predictor),
       GCG(GCG),
       rank_manager(rank_manager),
       initial_utilization_status(get_simulator()) { }
    
    // 主调度函数
    std::unordered_map<NodeID, Rank> 
    RunDynamicScheduling(const std::vector<NodeID> &nodes_to_schedule);

    size_t nr_node_acc_pair__to_traval;
    size_t nr_ranks;
    size_t nr_nodes_to_be_scheduled;
    double fast_heru_time;
    double time__to__traval_all_node_acc_pair;

private:
    std::function<Duration (AccModel, TaskID, TaskNodeID)> task_predictor;
    std::function<std::pair<Duration, NBytes> (Rank, Rank, std::shared_ptr<VariableDescriptor>)> transmit_predictor;
    std::shared_ptr<GCG_Adding_OpManagement> GCG;
    std::shared_ptr<RankManager> rank_manager;
    const UtilizationStatus initial_utilization_status;


    // 适配函数计算
    std::pair<double, Timestamp> CalculateOperatorScore(
                            SearchBranch &search_branch,
                            NodeID node_id, Rank rank,
                            bool update_utilization_status,
                            ScoreBreakdown* out_bd);

    // 候选节点生成实现
    std::vector<Rank>
    GetCandidateRanks(SearchBranch& branch,
                                    const std::vector<NodeID>& nodes_to_schedule,
                                    size_t k);

    double 
    CalculateTotalCost(SearchBranch &branch,
                       const std::vector<NodeID> &nodes_to_schedule);

    double 
    CalculateOptimisticLowerBound(SearchBranch &branch,
                                  const std::vector<NodeID> &nodes_to_schedule,
        ScoreBreakdown* out_bd = nullptr);



    void
    CompleteScheduleGreedily(
        SearchBranch& branch,
        const std::vector<NodeID>& nodes_to_schedule);


    // 计算未调度节点的关键路径长度
    double 
    CalculateCriticalPathLength(const std::vector<NodeID>& nodes);

    void
    Next_Op_Assign_To(
        SearchBranch &branch,
        NodeID node_id,
        Rank r
    );
};

#endif // DYNAMIC_SCHEDULER_H