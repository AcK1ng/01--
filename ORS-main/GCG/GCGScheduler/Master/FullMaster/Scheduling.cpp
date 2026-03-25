#include "Scheduling.h"
#include <fstream>
#include <iostream>
#include <functional>

#include "Scheduling.h"
#include <fstream>
#include <iostream>
#include <functional>
#include <algorithm>
#include <numeric>

struct SearchStats {
    long long total_branches_generated = 0;
    long long pruned_by_LB = 0;
    long long pruned_by_beam = 0;
    int max_depth_reached = 0;
    int iterations_run = 0;
};

std::unordered_map<NodeID, Rank> DynamicScheduler::RunDynamicScheduling(
    const std::vector<NodeID> &nodes_to_schedule
) {
    double UB = std::numeric_limits<double>::max();
    std::unordered_map<NodeID, Rank> initial_solution;

    SearchBranch initial_fast_solution;
    initial_fast_solution.utilization_status = this->initial_utilization_status;

    Timestamp s, e;
    size_t nr_ranks = rank_manager->GetAllRanks().size();
    size_t nr_nodes_to_be_scheduled = nodes_to_schedule.size();
    double fast_heru_time, time__to__traval_all_node_acc_pair = 0;
    size_t nr_node_acc_pair__to_traval = 0;

    s = RealTimeNow();
    this->CompleteScheduleGreedily(initial_fast_solution, nodes_to_schedule);
    UB = CalculateTotalCost(initial_fast_solution, nodes_to_schedule); 
    e = RealTimeNow();
    fast_heru_time = (double)(e - s) / 1000000;

    std::unordered_map<NodeID, Rank> scheduling_results = initial_fast_solution.placement;

    SearchStats stats;
    std::vector<SearchBranch> active_branches;
    SearchBranch initial_branch;
    initial_branch.utilization_status = this->initial_utilization_status;
    active_branches.push_back(initial_branch);

    const size_t k = 2; 
    const size_t max_iterations = 100;
    size_t iterations = 0;
    bool UB_updated = true;

    while (!active_branches.empty() && iterations < max_iterations) {
        std::vector<SearchBranch> new_branches;
        
        for (auto &branch : active_branches) {
            if (branch.node_idx__to_be_scheduled > stats.max_depth_reached) {
                stats.max_depth_reached = branch.node_idx__to_be_scheduled;
            }
            if (nodes_to_schedule.size() <= branch.node_idx__to_be_scheduled) continue;

            NodeID next_node = nodes_to_schedule[branch.node_idx__to_be_scheduled];
            std::vector<Rank> candidate_ranks = GetCandidateRanks(branch, nodes_to_schedule, k);
            
            for (Rank candidate_rank : candidate_ranks) {
                SearchBranch new_branch = branch;
                stats.total_branches_generated++; 
                nr_node_acc_pair__to_traval++;

                s = RealTimeNow();
                this->Next_Op_Assign_To(new_branch, next_node, candidate_rank);
                new_branch.LB = new_branch.cost_so_far;
                            //   + CalculateOptimisticLowerBound(new_branch, nodes_to_schedule, nullptr);
                e = RealTimeNow();
                time__to__traval_all_node_acc_pair += (double)(e - s) / 1000000;

                if (new_branch.LB > UB) {
                    // 剪枝
                    stats.pruned_by_LB++;
                    continue;
                }

                new_branches.push_back(new_branch);
            }
        }
        
        size_t branches_before_beam = new_branches.size();
        
        if (new_branches.size() > k) {
            std::sort(new_branches.begin(), new_branches.end(),
                    [](const SearchBranch& a, const SearchBranch& b) { return a.LB < b.LB; });
            new_branches.resize(k);
        }
        stats.pruned_by_beam += (branches_before_beam - new_branches.size());
        
        if (new_branches.size() >= 2) {
            for (size_t i = 0; i < 2; ++i) {
                SearchBranch greedy_solution = new_branches[i];
                this->CompleteScheduleGreedily(greedy_solution, nodes_to_schedule);
                double greedy_cost = CalculateTotalCost(greedy_solution, nodes_to_schedule); 
                
                if (greedy_cost < UB) {
                    UB = greedy_cost;
                    scheduling_results = greedy_solution.placement;
                    UB_updated = true;
                }
            }
        }
        
        if (!UB_updated && iterations > 10) break;
        
        active_branches = new_branches;
        iterations++;
        UB_updated = false;
    }

    this->nr_node_acc_pair__to_traval = nr_node_acc_pair__to_traval;
    this->nr_ranks = nr_ranks;
    this->nr_nodes_to_be_scheduled = nr_nodes_to_be_scheduled;
    this->fast_heru_time = fast_heru_time;
    this->time__to__traval_all_node_acc_pair = time__to__traval_all_node_acc_pair;

    return scheduling_results;
}


std::pair<double, Timestamp>
DynamicScheduler::CalculateOperatorScore(
    SearchBranch &search_branch,
    NodeID node_id, Rank rank,
    bool update_utilization_status,
    ScoreBreakdown *out_bd
) {
    struct Node *node = GCG->__get_node(node_id);
    UtilizationStatus &utilization_status = search_branch.utilization_status;

    double alpha = 1.0, beta = 1.0, gamma = 1.0; 
    // double mu = 0.1;

    ComputeDomain domain = rank_manager->GetComputeDomain(rank);
    auto acc_model = rank_manager->GetModelByHost(rank_manager->GetHostByRank(rank));

    Duration original_compute_time = this->task_predictor(acc_model, node->_.task_id, node->_.task_node_id);
    Duration base_compute_time = this->task_predictor(rank_manager->GetStandardModelByTaskNodeID(node->_.task_node_id),
                                                      node->_.task_id, node->_.task_node_id);
    if (base_compute_time == 0) base_compute_time = 1; 

    double normal_compute_time = (double)original_compute_time / base_compute_time;

    Duration original_joint_comm_time = 0;
    Duration base_joint_comm_time = 0;
    double normal_joint_comm_time;

    constexpr double INTRA_HOST_BANDWIDTH_BYTES_PER_SEC = 1099511627776.0; 
    constexpr double INTER_HOST_BANDWIDTH_BYTES_PER_SEC = 13421772800.0;
    constexpr double SECONDS_TO_NANOSECONDS = 1000000000.0;

    auto get_base_comm_time = [&](Rank s, Rank r, NBytes nbytes) -> Duration {
        HostID sh = this->rank_manager->GetHostByRank(s);
        HostID rh = this->rank_manager->GetHostByRank(r);
        
        double bandwidth;
        if (sh == rh) bandwidth = INTRA_HOST_BANDWIDTH_BYTES_PER_SEC;
        else bandwidth = INTER_HOST_BANDWIDTH_BYTES_PER_SEC;

        double time_in_seconds = (double)nbytes / bandwidth;
        Duration base_comm_time_ns = (Duration)(time_in_seconds * SECONDS_TO_NANOSECONDS);

        if (base_comm_time_ns == 0) return 1;
        return base_comm_time_ns;
    };

    Duration wait_time = 0;
    double balance_penalty = 0.0;
    Timestamp op_end_timestamp;

    if (update_utilization_status) { 
        auto CalculateWaitTime = [&]() -> Duration {
            bool has_prev_op = false;
            Timestamp available_time = utilization_status.rank__to__available_time[rank];
            Timestamp op_ready_time = 0;
            Timestamp op_prev = 0;

            for (NodeID input_id: node->inputs) {
                struct Node *input_node = GCG->__get_node(input_id);
                if (input_node->_.is_constant) continue;
                has_prev_op = true;

                bool is_scheduling_op = search_branch.placement.contains(input_id);
                if (is_scheduling_op)
                    op_prev = std::max(op_prev, search_branch.end_timestamps[input_id]);
                else if (input_node->_.done)
                    op_prev = std::max(op_prev, input_node->_.done_timestamp);
                else if (input_node->_.is_scheduled)
                    op_prev = std::max(op_prev, utilization_status.scheduled_op_end_time->at(input_id));
                else
                    assert(0);

                auto [transmit_duration, transmit_size] =
                    this->transmit_predictor(input_node->_.assigned_to, rank, input_node->_.shape);

                op_ready_time = utilization_status.transmit(input_node->_.assigned_to, rank, domain,
                                                            transmit_duration, transmit_size, is_scheduling_op);
                original_joint_comm_time += transmit_duration;
                base_joint_comm_time += get_base_comm_time(input_node->_.assigned_to, rank, transmit_size);
            }
            op_end_timestamp = utilization_status.compute(rank, domain, original_compute_time,
                                                          total_size_of_vd(node->_.shape));

            if (!has_prev_op) return 0;
            if (op_ready_time < available_time) return available_time - op_ready_time;
            return 0;
        };

        balance_penalty = utilization_status.CalculateUtilizationVariance();
        wait_time = CalculateWaitTime();
        balance_penalty = utilization_status.CalculateUtilizationVariance() - balance_penalty;
    } else {
        auto CalculateWaitTime = [&]() -> Duration {
            bool has_prev_op = false;
            Timestamp op_prev = 0;
            for (NodeID input_id: node->inputs) {
                struct Node *input_node = GCG->__get_node(input_id);
                if (input_node->_.is_constant) continue;
                has_prev_op = true;
                bool is_scheduling_op = search_branch.placement.contains(input_id);
                if (is_scheduling_op) op_prev = std::max(op_prev, search_branch.end_timestamps[input_id]);
                else if (input_node->_.done) op_prev = std::max(op_prev, input_node->_.done_timestamp);
                else if (input_node->_.is_scheduled) op_prev = std::max(op_prev, utilization_status.scheduled_op_end_time->at(input_id));
                else assert(0);
                auto [transmit_duration, transmit_size] = this->transmit_predictor(input_node->_.assigned_to, rank, input_node->_.shape);
                original_joint_comm_time += transmit_duration;
                base_joint_comm_time += get_base_comm_time(input_node->_.assigned_to, rank, transmit_size);
            }
            Timestamp op_ready_time = op_prev + original_joint_comm_time;
            Timestamp available_time = utilization_status.rank__to__available_time[rank];
            Timestamp op_start_time = std::max(op_ready_time, available_time);
            op_end_timestamp = op_start_time + original_compute_time;
            if (!has_prev_op) return 0;
            if (op_ready_time < available_time) return available_time - op_ready_time;
            return 0;
        };
        wait_time = CalculateWaitTime();
        balance_penalty = utilization_status.CalculateUtilizationVariance();
    }

    if (base_joint_comm_time == 0) normal_joint_comm_time = 0.0;
    else normal_joint_comm_time = (double)original_joint_comm_time / base_joint_comm_time;
    auto CalculateMemoryPenalty = [&]() -> double {
        size_t memory_capacity = rank_manager->GetHBMCapability(domain);
        size_t memory_usage = utilization_status.domain__to__hbm_usage[domain];
        if (memory_usage > memory_capacity) return std::numeric_limits<double>::infinity();
        double epsilon = 0.1;
        return (double)memory_usage / memory_capacity - epsilon;
    };
    double mem_penalty = CalculateMemoryPenalty();

    double normalized_wait_time = (double)wait_time / base_compute_time;

    double score = alpha * normal_compute_time + 
                   beta  * normal_joint_comm_time + 
                   gamma * normalized_wait_time + 
                   mem_penalty + 
                   balance_penalty;

    if (out_bd) {
        out_bd->raw_comp_ratio = normal_compute_time;
        out_bd->raw_comm_ratio = normal_joint_comm_time;
        out_bd->raw_wait_ratio = normalized_wait_time;
        out_bd->raw_mem_val    = mem_penalty;
        out_bd->raw_bal_val    = balance_penalty;
    }

    return {score, op_end_timestamp};
}


std::vector<Rank>
DynamicScheduler::GetCandidateRanks(SearchBranch& branch,
                                    const std::vector<NodeID>& nodes_to_schedule,
                                    size_t k) {
    std::vector<Rank> candidates;
    NodeID node_id = nodes_to_schedule[branch.node_idx__to_be_scheduled];
    bool is_first_operator = (branch.node_idx__to_be_scheduled == 0);
    
    if (is_first_operator) {
        auto all_ranks = rank_manager->GetAllRanks();
        std::vector<std::pair<Rank, double>> ranked_ranks;
        
        for (Rank rank : all_ranks) {
            auto [score, op_end_timestamp] = CalculateOperatorScore(branch, node_id, rank, false, nullptr);
            ranked_ranks.emplace_back(rank, score);
        }
        
        std::sort(ranked_ranks.begin(), ranked_ranks.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; });
        
        for (size_t i = 0; i < std::min(k, ranked_ranks.size()); ++i) {
            candidates.push_back(ranked_ranks[i].first);
        }
    } else {
        auto* node = GCG->__get_node(node_id);
        std::unordered_set<Rank> predecessor_ranks;
        for (NodeID input_id : node->inputs) {
            if (branch.placement.find(input_id) != branch.placement.end()) {
                predecessor_ranks.insert(branch.placement.at(input_id));
            }
        }
        
        for (Rank pred_rank : predecessor_ranks) {
            candidates.push_back(pred_rank); 
            auto neighbors = rank_manager->GetTopKNeighbors(pred_rank, 2 * k);
            candidates.insert(candidates.end(), neighbors.begin(), neighbors.end());
        }
        
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        
        if (candidates.size() > 2 * k) {
            candidates.resize(2 * k);
        }
    }
    return candidates;
}

double DynamicScheduler::CalculateTotalCost(
    SearchBranch &branch,
    const std::vector<NodeID> &nodes_to_schedule
) {
    for (size_t i = branch.node_idx__to_be_scheduled; i < nodes_to_schedule.size(); i++) {
        NodeID node_id = nodes_to_schedule[i];
        Rank rank = branch.placement[node_id];

        this->Next_Op_Assign_To(branch, node_id, rank);
    }

    return branch.cost_so_far;
}

// ==========================================
// 3. CalculateOptimisticLowerBound (LB 计算)
// ==========================================
// double
// DynamicScheduler::CalculateOptimisticLowerBound(
//     SearchBranch &branch,
//     const std::vector<NodeID> &nodes_to_schedule,
//     ScoreBreakdown* out_bd // <--- 1. 只改了参数列表，去掉了默认值
// ) {

//     size_t nr_unscheduled_nodes = nodes_to_schedule.size() - branch.node_idx__to_be_scheduled;

//     if (nr_unscheduled_nodes == 0)
//         return 0.0;
    
//     // 计算未调度节点的乐观估计
//     // 1. 计算最快的计算时间（假设使用最快的设备）
//     double min_compute_time = std::numeric_limits<double>::max();
//     auto all_ranks = rank_manager->GetAllRanks();
    
//     for (size_t i = branch.node_idx__to_be_scheduled; i < nodes_to_schedule.size(); i++) {
//         NodeID node_id = nodes_to_schedule[i];

//         auto* node = GCG->__get_node(node_id);

//         //设置base_compute_time
//         Duration base_compute_time = this->task_predictor(
//             rank_manager->GetStandardModelByTaskNodeID(node->_.task_node_id),
//             node->_.task_id, node->_.task_node_id);
        
//         if (base_compute_time == 0) base_compute_time = 1; // 防除0
        
//         // 找到最快的设备计算时间
//         for (Rank rank : all_ranks) {
//             auto acc_model = rank_manager->GetModelByHost(rank_manager->GetHostByRank(rank));
//             Duration compute_time = (double) this->task_predictor(acc_model, node->_.task_id, node->_.task_node_id) / base_compute_time;
            
//             if (compute_time < min_compute_time) {
//                 min_compute_time = static_cast<double>(compute_time);
//             }
//         }
//     }

//     // 2. 估算通信时间
//     double min_comm_time = 0.0;
//     // 这里可以进一步优化，但为了简单起见，我们使用一个固定的小值
//     min_comm_time = 0.1 * min_compute_time * nr_unscheduled_nodes;
    
//     // 3. 乐观估计：假设所有未调度节点可以完全并行执行
//     // 使用关键路径长度而不是简单求和

//     // 简单的关键路径估算
//     double critical_path_length = std::sqrt(static_cast<double>(nr_unscheduled_nodes));
    
//     // 计算最终的计算部分贡献
//     double total_compute_part = critical_path_length * min_compute_time;

//     // === [新增] 记录 Breakdown ===
//     // 逻辑：完全照搬你原本的返回值构成
//     if (out_bd) {
//         out_bd->raw_comp_ratio = total_compute_part; // 计算部分 = 关键路径 * 最小时间
//         out_bd->raw_comm_ratio = min_comm_time;      // 通信部分
//         out_bd->raw_wait_ratio = 0.0;                // LB 乐观估计无等待
//         out_bd->raw_mem_val    = 0.0;
//         out_bd->raw_bal_val    = 0.0;
//     }
//     // ============================
    
//     return total_compute_part + min_comm_time; //即最长路径上的节点数，或者更精确的是依赖路径上的计算时间总和的最小可能值，这里我们用节点数乘以最快计算时间来表示
// }


void
DynamicScheduler::CompleteScheduleGreedily(
    SearchBranch& branch,
    const std::vector<NodeID>& nodes_to_schedule) {
    while (branch.node_idx__to_be_scheduled < nodes_to_schedule.size()) {
        NodeID next_node = nodes_to_schedule[branch.node_idx__to_be_scheduled];
        Rank best_rank = -1;
        double best_score = std::numeric_limits<double>::max();
        Timestamp best_op_end_timestamp = 0.0;
        
        auto all_ranks = rank_manager->GetAllRanks();
        for (Rank rank : all_ranks) {
            auto [score, op_end_timestamp] = CalculateOperatorScore(branch, next_node, rank, false, nullptr);
            if (score < best_score) {
                best_score = score;
                best_rank = rank;
                best_op_end_timestamp = op_end_timestamp;
            }
        }

        branch.placement[next_node] = best_rank;
        branch.end_timestamps[next_node] = best_op_end_timestamp;
    }
}

void
DynamicScheduler::Next_Op_Assign_To(
    SearchBranch &branch,
    NodeID node_id,
    Rank r
) {
#ifdef TEST_SCHEDULING_ALGORITHM
    branch.debug_breakdowns.emplace_back();
#endif

    auto [score, op_end_timestamp] = this->CalculateOperatorScore(
        branch, node_id, r, true,
#ifdef TEST_SCHEDULING_ALGORITHM
        &(branch.debug_breakdowns.back())
#else
        nullptr
#endif
    );

    branch.placement[node_id] = r;
    branch.cost_so_far += score;
    branch.end_timestamps[node_id] = op_end_timestamp;
    branch.node_idx__to_be_scheduled++;
}