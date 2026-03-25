#ifndef CLUSTER_STATUS_H
#define CLUSTER_STATUS_H
#include "Base.h"
#include "Master/FullMaster/GCG.h"
#include "Master/FullMaster/RankManager.h"
#include "Master/FullMaster/TransmitManager.h"
#include "Master/FullMaster/PerfPredictor.h"

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


std::mutex debug_output_mutex;

class FakeRegPayload;
class FakeComputeDomain {
public:
    FakeComputeDomain(size_t hbm_capability):
        hbm_capability(hbm_capability), used_hbm(0) { }
    inline void malloc(size_t size) {this->used_hbm += size;}
    inline void free(size_t size) {this->used_hbm -= size;}
    std::unordered_map<TransmitID, std::shared_ptr<FakeRegPayload>> node_to_be_transmit__inner_domain;
    size_t hbm_capability;
    size_t used_hbm;
};


class FakeRegPayload {
public:
    FakeRegPayload() = delete;
    FakeRegPayload(FakeComputeDomain *compute_domain, size_t size) {
        this->compute_domain = compute_domain;
        this->size = size;
        this->compute_domain->malloc(size);
    }
    ~FakeRegPayload() {
        this->compute_domain->free(size);
    }
    
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(FakeRegPayload,
        size
    )
private:
    FakeComputeDomain *compute_domain;
    size_t size;
};


class ActionForSimulation {
public:
    std::shared_ptr<const AccActionSpec> acc_action_spec;

    // We don't want to touch 'wanted_trace' in acc_action_spec.
    // 所以在这里再开一个wanted_trace来标记trace是否已经满足了
    std::optional<Trace> wanted_trace;
    std::shared_ptr<FakeRegPayload> wanted_trace_payload; // 拿到trace后，这里就得有东西了

    std::vector<std::shared_ptr<FakeRegPayload>> inputs;

    std::shared_ptr<FakeRegPayload> will_output; // Action执行中持有，执行完成后输出出去的payload
    std::shared_ptr<FakeRegPayload> will_leave_as_trace; // Action执行中持有，要留下trace的话就会留下这个

    std::optional<IssuingID> issuing_id;
    Rank rank;


    // 这里的done follower为了处理send和recv的配对
    // recv先执行，但是recv的执行完毕是由send触发的
    // 所以recv就是send的done follower
    Timestamp start;
    bool is_done_follower;

    // 这两个字段专门为 report_recv_done 设计的
    NBytes output_size;
    Duration transmit_duration;

    Timestamp predicted_end;
    std::shared_ptr<ActionForSimulation> done_follower;

    // 这个json，只是作为打印用，所以不用管那些FakeRegPayload具体怎么样的
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(ActionForSimulation,
        acc_action_spec,
        wanted_trace,
        wanted_trace_payload,
        inputs,
        will_output,
        will_leave_as_trace,
        issuing_id,
        rank,
        start,
        is_done_follower
    )

    ActionForSimulation(const ActionForSimulation &) = delete;

    ActionForSimulation(std::optional<IssuingID> issuing_id,
                        std::shared_ptr<const AccActionSpec> acc_action_spec):
        acc_action_spec(acc_action_spec),
        inputs(acc_action_spec->input_ids.size()),
        issuing_id(issuing_id),
        is_done_follower(0),
        done_follower(nullptr),
        wanted_trace(this->acc_action_spec->wanted_trace) {
        this->__update_ready();
    }

    ActionForSimulation(std::optional<IssuingID> issuing_id,
                        int priority,
                        size_t master_hint_sequence,
                        size_t action_commit_id,
                        AccActionEnum op_enum,
                        std::shared_ptr<AccActionParamPayload> op_param,
                        std::vector<NodeID> input_ids,
                        std::optional<NodeID> output_id,
                        std::optional<Trace> trace_will_leave,
                        StreamID stream_id,
                        bool no_OOM,
                        std::optional<Trace> wanted_trace,
                        bool clear_trace):
        inputs(input_ids.size()),
        issuing_id(issuing_id),
        is_done_follower(0),
        done_follower(nullptr) {
        this->acc_action_spec = std::make_shared<const AccActionSpec>(
            priority, master_hint_sequence, action_commit_id,
            op_enum, op_param, input_ids, output_id, trace_will_leave,
            stream_id, no_OOM, wanted_trace, clear_trace);

        this->wanted_trace = this->acc_action_spec->wanted_trace;

        this->__update_ready();
    }

    ActionForSimulation(std::optional<IssuingID> issuing_id,
                        AccActionSpec &opspec):
        acc_action_spec(std::make_shared<const AccActionSpec>(opspec)),
        inputs(this->acc_action_spec->input_ids.size()),
        issuing_id(issuing_id),
        is_done_follower(0),
        done_follower(nullptr),
        wanted_trace(this->acc_action_spec->wanted_trace) {
        this->__update_ready();
    }

    bool provide_trace(Trace trace, std::shared_ptr<FakeRegPayload> trace_payload) {
        if (this->wanted_trace.has_value()
                && this->wanted_trace.value() == trace) {
            this->wanted_trace = std::nullopt;
            this->wanted_trace_payload = trace_payload;
            this->__update_ready();
            if (this->acc_action_spec->clear_trace)
                return true; // the trace is cleared
            else
                return false;
        }
        return false;
    }

    void provide_input(NodeID reg_id, std::shared_ptr<FakeRegPayload> reg_payload) {
        int need_update = 0;
        for (size_t i = 0; i < this->acc_action_spec->input_ids.size(); i++)
            if (this->acc_action_spec->input_ids[i] == reg_id) {
                this->inputs[i] = reg_payload;
                need_update = 1;
            }
        if (need_update)
            this->__update_ready();
    }

    bool ready() { return this->__cached_ready; }
private:
    bool __cached_ready;
    void __update_ready() {
        this->__cached_ready = (this->wanted_trace == std::nullopt) && (this->__all_input_ready());
    }
    bool __all_input_ready() {
        return std::all_of(this->inputs.cbegin(),
                           this->inputs.cend(),
                           [](std::shared_ptr<FakeRegPayload> payload) { return payload ? true : false; });
    }
};

std::ostream&
operator<<(std::ostream &os, const struct ActionForSimulation &action) {
    nlohmann::json action_j = action;
    os << action_j.dump();
    return os;
}


auto running_action_cmp = [] (std::shared_ptr<ActionForSimulation> &l,
                              std::shared_ptr<ActionForSimulation> &r) -> bool {
    // 构造最小堆。predicted_end越小，越优先出队
    return l->predicted_end > r->predicted_end;
};

auto ready_action_cmp = [] (std::shared_ptr<ActionForSimulation> &l,
                            std::shared_ptr<ActionForSimulation> &r) -> bool {
    return l->acc_action_spec->acc_action_base < r->acc_action_spec->acc_action_base;
};



class FakeAcc {
public:
    FakeAcc(std::shared_ptr<FakeComputeDomain> compute_domain):
        compute_domain(compute_domain), stream_busy(NR_STREAM, false) {
        for (StreamID stream_id = 0; stream_id < NR_STREAM; stream_id++)
            this->streams.emplace_back(ready_action_cmp);
    }

    inline std::shared_ptr<FakeRegPayload>
    malloc(size_t size) {
        return std::make_shared<FakeRegPayload>(this->compute_domain.get(), size);
    }

    std::shared_ptr<FakeComputeDomain> compute_domain;

    std::vector<bool> stream_busy; // StreamID -> bool, 指这一瞬间是否有action正在运行
    std::vector<my_priority_queue<std::shared_ptr<ActionForSimulation>>> streams; // StreamID -> ready queue
    std::unordered_map<std::optional<IssuingID>,
                       std::list<std::shared_ptr<ActionForSimulation>>> queues; // IssuingID -> non-ready queue

    std::multiset<Trace> traces__will_be_left; // 这个意思是，有多少action生成trace，还没生成呢
    std::multiset<Trace> traces__will_be_deleted; // 这个意思是，那些trace生成了就立刻删掉，还没生成呢，如果action完成后发现这里有它生成的trace，就直接把trace扔掉

    std::multiset<Trace> traces_in_queue;
    std::unordered_map<NodeID, std::shared_ptr<FakeRegPayload>> trace_payloads; // Checkpoint_id -> fake payload

    inline bool
    __enqueue_action(std::shared_ptr<ActionForSimulation> action) {
        if (action->wanted_trace.has_value()) {
            auto trace = action->wanted_trace.value();
            auto it = this->traces_in_queue.find(trace);
            if (it != this->traces_in_queue.end()) {
                std::optional<NodeID> &ckpt_id = std::get<0>(trace);
                std::shared_ptr<FakeRegPayload> fake_payload = nullptr;
                if (ckpt_id.has_value())
                    fake_payload = this->trace_payloads.at(ckpt_id.value());
                action->provide_trace(trace, fake_payload);
                if (action->acc_action_spec->clear_trace) {
                    this->traces_in_queue.erase(it);
                    if (ckpt_id.has_value())
                        this->trace_payloads.erase(ckpt_id.value());
                }
            }
        }

        if (action->acc_action_spec->trace_will_leave.has_value())
            this->traces__will_be_left.insert(action->acc_action_spec->trace_will_leave.value());

        std::optional<IssuingID> &issuing_id = action->issuing_id;
        if (action->ready()) {
            StreamID stream_id = action->acc_action_spec->stream_id;
            this->streams[stream_id].push(action);
            return true;
        } else {
            if (!this->queues.contains(issuing_id))
                this->queues[issuing_id];
            this->queues[issuing_id].push_back(action);
            return false;
        }
    }

    inline void
    __purge_trace(Trace trace) {
        std::optional<NodeID> &ckpt_id = std::get<0>(trace);
        this->traces_in_queue.erase(trace);
        if (ckpt_id.has_value())
            this->trace_payloads.erase(ckpt_id.value());
        size_t nr_traces_will_gen = this->traces__will_be_left.count(trace);
        size_t nr_traces_will_be_delete = this->traces__will_be_deleted.count(trace);
        for (size_t i = 0; i < nr_traces_will_gen - nr_traces_will_be_delete; i++)
            this->traces__will_be_deleted.insert(trace);
    }

    inline void
    __broadcast__to_non_ready_list(const std::optional<IssuingID> &_issuing_id,
                                   const std::optional<NodeID> &output_id_,
                                   std::shared_ptr<FakeRegPayload> reg_payload,
                                   const std::optional<Trace> &trace_left_,
                                   std::shared_ptr<FakeRegPayload> trace_payload) {
        // non-ready queue，出队
        // ready queue，入队

        std::vector<std::optional<IssuingID>> issuing_ids = {std::nullopt, -1};
        if (_issuing_id.has_value())
            issuing_ids.push_back(_issuing_id.value());
        
        if (output_id_.has_value()) {
            for (auto &issuing_id: issuing_ids) {
                if (!this->queues.contains(issuing_id))
                    continue;
                
                auto &non_ready_list = this->queues[issuing_id];
                auto output_id = output_id_.value();
                auto it = non_ready_list.begin();
                while (it != non_ready_list.end()) {
                    (*it)->provide_input(output_id, reg_payload);
                    if ((*it)->ready()) {
                        this->streams[(*it)->acc_action_spec->stream_id].push(*it);
                        non_ready_list.erase(it++);
                    } else
                        it++;
                }
            }
        }

        do {
            if (!trace_left_.has_value())
                break;

            if (auto it = this->traces__will_be_deleted.find(trace_left_.value());
                it != this->traces__will_be_deleted.end()) {
                this->traces__will_be_deleted.erase(it);
                break;
            }

            auto trace_left = trace_left_.value();
            bool trace_gone = false;
            for (auto &p: this->queues) {
                auto &non_ready_list = p.second;
                auto it = non_ready_list.begin();
                while (it != non_ready_list.end()) {
                    trace_gone = (*it)->provide_trace(trace_left, trace_payload);
                    if ((*it)->ready()) {
                        this->streams[(*it)->acc_action_spec->stream_id].push(*it);
                        non_ready_list.erase(it++);
                    } else
                        it++;
                    if (trace_gone)
                        break;
                }
                if (trace_gone)
                    break;
            }
            
            if (!trace_gone) {
                // nobody needs the trace
                std::optional<NodeID> &ckpt_id = std::get<0>(trace_left);
                this->traces_in_queue.insert(trace_left);
                if (ckpt_id.has_value())
                    this->trace_payloads.insert({ckpt_id.value(), trace_payload});
            }
        } while(0);

        for (auto &issuing_id: issuing_ids) {
            if (this->queues.contains(issuing_id) && this->queues[issuing_id].empty())
                this->queues.erase(issuing_id);
        }
    }
    inline void
    __report_action_done(std::shared_ptr<ActionForSimulation> &last_done_action) {
        StreamID stream_id = last_done_action->acc_action_spec->stream_id;
        assert(this->stream_busy[stream_id]);
        this->stream_busy[stream_id] = false;
        if (last_done_action->acc_action_spec->trace_will_leave.has_value()) {
            if (auto it = this->traces__will_be_left.find(last_done_action->acc_action_spec->trace_will_leave.value());
                it != this->traces__will_be_left.end())
                this->traces__will_be_left.erase(it); // 删掉一个trace
        }
    }

    inline std::shared_ptr<ActionForSimulation>
    __fetch_next_running_action__when_idle(StreamID stream_id) {
        if (this->stream_busy[stream_id])
            return nullptr;
        if (this->streams[stream_id].empty())
            return nullptr;
        // ready queue 出队
        auto start_action = this->streams[stream_id].top();
        this->streams[stream_id].pop();
        this->stream_busy[stream_id] = true;
        return start_action;
    }

    inline void
    __before_running_action(std::shared_ptr<ActionForSimulation> action,
                            std::function<std::shared_ptr<VariableDescriptor> (NodeID)> ask_node_shape) {
        // 开始运行，为了运行action分配HBM
        switch (action->acc_action_spec->op_enum) {
        case HelloWorld:
        case InitATenRuntime:
        case DummyOutput: {
            } break;
        case RunATenOP:
        case UploadTensor: {
            size_t node_size = total_size_of_vd(ask_node_shape(action->acc_action_spec->output_id.value()));
            action->will_output = this->malloc(node_size);
            } break;
        case Transmit_RequestSendToMaster: {
            auto param = std::static_pointer_cast<struct TransmitInfo>(action->acc_action_spec->op_param);
            assert(param->send_domain != param->recv_domain);
            } break;
        case Transmit_AllocTensor: {
            auto param = std::static_pointer_cast<struct TransmitInfo>(action->acc_action_spec->op_param);
            assert(param->send_domain != param->recv_domain);
            size_t node_size = total_size_of_vd(ask_node_shape(param->node__to_transmit));
            action->will_leave_as_trace = this->malloc(node_size);
            } break;
        case Transmit_Recv: {
            auto param = std::static_pointer_cast<struct TransmitInfo>(action->acc_action_spec->op_param);
            if (param->send_domain == param->recv_domain) {
                TransmitID transmit_id = param->transmit_id;
                action->will_output = this->compute_domain->node_to_be_transmit__inner_domain.at(transmit_id);
                this->compute_domain->node_to_be_transmit__inner_domain.erase(transmit_id);
            } else {
                action->will_output = action->wanted_trace_payload;
            }
            } break;
        case Transmit_Send: {
            auto param = std::static_pointer_cast<struct TransmitInfo>(action->acc_action_spec->op_param);
            if (param->send_domain == param->recv_domain) {
                TransmitID transmit_id = param->transmit_id;
                this->compute_domain->node_to_be_transmit__inner_domain[transmit_id] = action->inputs[0];
            }
            } break;
        case SettledAsCheckpoint: {
            action->will_leave_as_trace = action->inputs[0];
            } break;
        case FetchCheckpoint: {
            action->will_output = action->wanted_trace_payload;
            } break;
        }
    }
};


class ClusterStatusImpl: public Simulator::ClusterStatus {
private:
    std::shared_ptr<Master> master;
    std::shared_ptr<RankManager> rank_manager;
    my_priority_queue<std::shared_ptr<ActionForSimulation>> running_actions;
    std::unordered_map<TransmitID, std::shared_ptr<ActionForSimulation>> send_actions;

    std::unordered_map<Rank, FakeAcc> accs;
    std::unordered_map<ComputeDomain, std::shared_ptr<FakeComputeDomain>> fake_compute_domains;
    std::mutex lock;

#ifndef NDEBUG
    // 如果一个 Action a (recv) 的执行完毕需要以 Action b (send) 的执行完毕为条件，那么 a 是 b 的 done follower。
    // 就是说 a 得跟在 b 之后， b 的 done follower 是 a ； a 是 b 的 done leader。
    //
    // done follower 的麻烦之处在于，
    // b 在 running ；而 a 没有 running 的时候，不能在 running list 里找到 b 。
    // 在debug的时候可能有点麻烦，所以在running list没有b的时候，先把b加到下面这个done_followers里。
    // 也就是说，done_followers + running list才是正在running的action的全集。
    std::unordered_set<std::shared_ptr<ActionForSimulation>> done_followers;
#endif

    virtual void
    check() override {
        std::lock_guard<std::mutex> guard(debug_output_mutex);
        if (this->__lint() == false) {
            this->debugOutput();
            assert(0);
        }
    }

    bool
    __lint() {
        for (auto &p: this->accs) {
            Rank rank = p.first;
            auto &acc = p.second;

            for (StreamID stream_id = 0; stream_id < NR_STREAM; stream_id++) {
                if (acc.stream_busy[stream_id]) {
#ifndef NDEBUG
                    {
                        size_t nr_of_running_op = 0;
                        for (auto action: this->running_actions.heap) {
                            if (action->rank == rank && action->acc_action_spec->stream_id == stream_id)
                                nr_of_running_op++;
                        }

                        for (auto action: this->done_followers) {
                            if (action->rank == rank && action->acc_action_spec->stream_id == stream_id)
                                nr_of_running_op++;
                        }
                        
                        if (nr_of_running_op != 1) {
                            std::cout << "Lint Error: Rank[" << rank << "] StreamID[" << stream_id << "] "
                                      << "stream_busy but nr_of_running_op=" << nr_of_running_op << std::endl;
                            return false;
                        }
                    }
#endif
                }

                auto &ready_queue = acc.streams[stream_id];

                if (!ready_queue.empty() && !acc.stream_busy[stream_id]) {
                    std::cout << "Lint Error: Rank[" << rank << "] StreamID[" << stream_id << "] "
                              << "ready_queue has op but stream is idle" << std::endl;
                    return false;
                }

                for (auto action: ready_queue.heap) {
                    if (!action->ready()) {
                        std::cout << "Lint Error: Rank[" << rank << "] StreamID[" << stream_id << "] "
                                  << "stream has non-ready action " << *action << std::endl;
                        return false;
                    }
                }
            }

            for (auto &p: acc.queues) {
                auto &non_ready_list = p.second;
                for (auto action: non_ready_list) {
                    if (action->ready()) {
                        std::cout << "Lint Error: Rank[" << rank << "] "
                                  << "non-ready queue has ready action " << *action << std::endl;
                        return false;
                    }
                }
            }
        }
        return true;
    }

public:

    virtual void
    IssueActions(Rank rank,
                 std::optional<IssuingID> issuing_id,
                 std::vector<std::shared_ptr<AccActionSpec>> &action_specs) override {
        // Issuing actions only because "Step" / "StartUp" / Master actively (User Submit)
        // 因 "Step" 和 因 Master actively 间的 IssueActions 要互斥

        auto &acc = this->accs.at(rank);

        // ready and non-ready  queue，入队

        this->lock.lock();
        for (auto action_: action_specs) {
            auto action = std::make_shared<ActionForSimulation>(issuing_id, *action_);
            action->rank = rank;

            if (action->acc_action_spec->op_enum == Transmit_Send) {
                // 处理 recv 和 send 的配对
                auto param = std::static_pointer_cast<struct TransmitInfo>(action->acc_action_spec->op_param);
                this->send_actions[param->transmit_id] = action;
            }

#ifndef NDEBUG
            // std::cout << "Simulator Timestamp[" << this->now << "] "
            //           << "Rank[" << rank << "] "
            //           << "Issue Action " << action->acc_action_spec << std::endl;
#endif

            acc.__enqueue_action(action);
            if (action->ready()) {
                StreamID stream_id = action->acc_action_spec->stream_id;
                this->__start_ready_action__for_idle_queue(rank, stream_id);
            }
        }
        this->lock.unlock();
    }

    virtual void
    FreeCheckpoint(Rank rank, NodeID node_id) override {
        this->lock.lock();
        auto &acc = this->accs.at(rank);
        acc.__purge_trace(Trace({node_id, std::nullopt}));
        this->lock.unlock();
    }

    virtual void
    PermitRecvs(Rank recv_rank,
                std::vector<std::shared_ptr<TransmitSpec>> &transmit_specs) override {
        // ready and non-ready  queue，入队
        this->lock.lock();
        for (auto transmit_spec: transmit_specs) {
            this->__broadcast__to_non_ready_list(recv_rank, transmit_spec->hint_issuing_id,
                                                 std::nullopt, nullptr,
                                                 Trace({std::nullopt, transmit_spec->transmit_id}), nullptr);
        }
        for (StreamID stream_id = 0; stream_id < NR_STREAM; stream_id++)
            this->__start_ready_action__for_idle_queue(recv_rank, stream_id);
        this->lock.unlock();
    }

    virtual Timestamp
    NextActionEndTimestamp() override {
        this->lock.lock();
        if (this->running_actions.empty()) {

#ifndef NDEBUG
            this->check();
#endif

            this->lock.unlock();
            return -1;
        }
        this->lock.unlock();
        return this->running_actions.top()->predicted_end;
    }

    virtual StepResult
    StepOneMoreActionEnd() override {
        this->lock.lock();

        auto expiring_action = this->running_actions.top();
        this->running_actions.pop();

        Rank rank = expiring_action->rank;
        StreamID stream_id = expiring_action->acc_action_spec->stream_id;
        auto &acc = this->accs.at(rank);

        this->now = expiring_action->predicted_end;

        acc.__report_action_done(expiring_action);
        this->__broadcast__to_non_ready_list(rank,
                                             expiring_action->issuing_id,
                                             expiring_action->acc_action_spec->output_id,
                                             expiring_action->will_output,
                                             expiring_action->acc_action_spec->trace_will_leave,
                                             expiring_action->will_leave_as_trace);

        for (StreamID stream_id = 0; stream_id < NR_STREAM; stream_id++)
            this->__start_ready_action__for_idle_queue(rank, stream_id);
            
        this->action_done_callback(rank,
            expiring_action->acc_action_spec,
            expiring_action->predicted_end,
            expiring_action->predicted_end - expiring_action->start);

        this->lock.unlock();

        this->__action_bottomhalf(expiring_action);

        return ExecuteAction;
    }

private:
    inline void
    __action_start(std::shared_ptr<ActionForSimulation> action) {
        // assert this->lock is unlocked, so that potential IssueActions/PermitRecvs can be lock
        switch (action->acc_action_spec->op_enum) {
        case HelloWorld: {
            } break;
        case InitATenRuntime: {
            } break;
        case RunATenOP: {
            } break;
        case UploadTensor: {
            } break;
        case Transmit_RequestSendToMaster: {
            } break;
        case Transmit_AllocTensor: {
            } break;
        case Transmit_Recv: {
            auto param = std::static_pointer_cast<struct TransmitInfo>(action->acc_action_spec->op_param);
            if (param->send_domain != param->recv_domain) {
                this->lock.lock();
                this->__broadcast__to_non_ready_list(param->send_rank, action->issuing_id,
                                                     std::nullopt, nullptr,
                                                     Trace({std::nullopt, param->transmit_id}), nullptr);
                this->__start_ready_action__for_idle_queue(param->send_rank, 0);
                this->lock.unlock();
            }
            } break;
        case Transmit_Send: {
            auto param = std::static_pointer_cast<struct TransmitInfo>(action->acc_action_spec->op_param);
            if (param->send_domain == param->recv_domain) {
                this->lock.lock();
                this->__broadcast__to_non_ready_list(param->recv_rank, action->issuing_id,
                                                     std::nullopt, nullptr,
                                                     Trace({std::nullopt, param->transmit_id}), nullptr);
                this->__start_ready_action__for_idle_queue(param->recv_rank, 1);
                this->lock.unlock();
            }
            } break;
        case SettledAsCheckpoint: {
            } break;
        case FetchCheckpoint: {
            } break;
        case DummyOutput: {
            } break;
        };
    }

    inline void
    __action_bottomhalf(std::shared_ptr<ActionForSimulation> action) {
        // assert this->lock is unlocked, so that potential IssueActions/PermitRecvs can be lock

        switch (action->acc_action_spec->op_enum) {
        case HelloWorld: {
            } break;
        case InitATenRuntime: {
            } break;
        case RunATenOP: {
            auto param = std::static_pointer_cast<struct RunATenOP>(action->acc_action_spec->op_param);
            auto param_to_master = std::make_shared<struct AccReportNodeDone>();
            param_to_master->node_id = param->node_id;
            param_to_master->task_id = param->task_id;
            param_to_master->task_node_id = param->task_node_id;
            param_to_master->running_time = action->predicted_end - action->start;
            param_to_master->end = action->predicted_end;
            this->SendAccEvent_ToMaster(action->rank, action->issuing_id, this->now,
                                        AccReportNodeDone, param_to_master);
            } break;
        case UploadTensor: {
            auto param = std::static_pointer_cast<struct UploadTensor>(action->acc_action_spec->op_param);
            auto param_to_master = std::make_shared<struct AccReportNodeDone>();
            param_to_master->node_id = param->node_id;
            param_to_master->running_time = 0;
            param_to_master->end = action->predicted_end;
            this->SendAccEvent_ToMaster(action->rank, action->issuing_id, this->now,
                                        AccReportNodeDone, param_to_master);
            } break;
        case Transmit_RequestSendToMaster: {
            auto param = std::static_pointer_cast<struct TransmitInfo>(action->acc_action_spec->op_param);
            auto param_to_master = std::make_shared<struct AccRequestSend>();
            param_to_master->transmit_id = param->transmit_id;
            param_to_master->recv_rank = param->recv_rank;
            param_to_master->variable_descriptor = this->ask_node_shape(param->node__to_transmit);
            this->SendAccEvent_ToMaster(action->rank, action->issuing_id, this->now,
                                        AccRequestSend, param_to_master);
            } break;
        case Transmit_AllocTensor: {
            auto param = std::static_pointer_cast<struct TransmitInfo>(action->acc_action_spec->op_param);
            } break;
        case Transmit_Recv: {
            auto param = std::static_pointer_cast<struct TransmitInfo>(action->acc_action_spec->op_param);
            auto param_to_master = std::make_shared<struct AccReportRecvDone>();
            param_to_master->transmit_id = param->transmit_id;
            param_to_master->profiling.push_back({action->output_size, action->transmit_duration});
            this->SendAccEvent_ToMaster(action->rank, action->issuing_id, this->now,
                                        AccReportRecvDone, param_to_master);
            } break;
        case Transmit_Send: {
            } break;
        case SettledAsCheckpoint: {
            auto param = std::static_pointer_cast<struct SettledAsCheckpoint>(action->acc_action_spec->op_param);
            auto param_to_master = std::make_shared<struct AccReportCheckpointSettled>();
            param_to_master->node_id = param->node_id;
            this->SendAccEvent_ToMaster(action->rank, action->issuing_id, this->now,
                                        AccReportCheckpointSettled, param_to_master);
            } break;
        case FetchCheckpoint: {
            } break;
        case DummyOutput: {
            } break;
        }
    }

    inline void
    __broadcast__to_non_ready_list(Rank rank,
                                   std::optional<IssuingID> issuing_id,
                                   std::optional<NodeID> output_id_,
                                   std::shared_ptr<FakeRegPayload> reg_payload,
                                   std::optional<Trace> trace_left_,
                                   std::shared_ptr<FakeRegPayload> trace_payload) {
        // assert this->lock is locked

        auto &acc = this->accs.at(rank);
        acc.__broadcast__to_non_ready_list(issuing_id, output_id_, reg_payload, trace_left_, trace_payload);

        // assert this->lock is locked
    }

    inline void
    __start_ready_action__for_idle_queue(Rank rank, StreamID stream_id) {
        auto &acc = this->accs.at(rank);

        // assert this->lock is locked

        // running_actions，入队

        auto start_action = acc.__fetch_next_running_action__when_idle(stream_id);
        if (start_action == nullptr)
            return;

        acc.__before_running_action(start_action, this->ask_node_shape);

        start_action->start = this->now;
        if (start_action->acc_action_spec->op_enum == RunATenOP) {
            auto param = std::static_pointer_cast<struct RunATenOP>(start_action->acc_action_spec->op_param);
            start_action->predicted_end = start_action->start
                                        + this->task_predictor(
                                              this->rank_manager->GetModelByHost(this->rank_manager->GetHostByRank(rank)),
                                              param->task_id,  
                                              param->task_node_id);
        } else if (start_action->acc_action_spec->op_enum == Transmit_Recv) {
            auto param = std::static_pointer_cast<struct TransmitInfo>(start_action->acc_action_spec->op_param);

            if (param->send_domain == param->recv_domain) {
                start_action->predicted_end = start_action->start;
            } else {
                auto send_action = this->send_actions[param->transmit_id];
                start_action->output_size = total_size_of_vd(param->variable_descriptor);
                start_action->transmit_duration = this->transmit_predictor(param->send_rank,
                                                                           param->recv_rank,
                                                                           param->variable_descriptor);
                                                                           
                this->send_actions.erase(param->transmit_id);

                start_action->is_done_follower = 1;
                send_action->done_follower = start_action;
            }

        } else if (start_action->acc_action_spec->op_enum == Transmit_Send) {
            auto param = std::static_pointer_cast<struct TransmitInfo>(start_action->acc_action_spec->op_param);
            if (param->send_domain == param->recv_domain) {
                start_action->predicted_end = start_action->start;
            } else {
                start_action->predicted_end = start_action->start
                                            + start_action->done_follower->transmit_duration;
            }
        } else
            start_action->predicted_end = start_action->start;

        if (start_action->done_follower) {
            start_action->done_follower->predicted_end = start_action->predicted_end;
            this->running_actions.push(start_action->done_follower);

#ifndef NDEBUG
            this->done_followers.erase(start_action->done_follower);
#endif

        }

        if (start_action->is_done_follower) {
            // 虽然这个action正在执行，但是它的done由它的done leader触发
            // 所以在这里不入running_actions队
            // 而且这个action的predicted_end先不用

#ifndef NDEBUG
            this->done_followers.insert(start_action);
#endif

        } else
            this->running_actions.push(start_action);

        this->lock.unlock();

        // ready and non-ready  queue，入队
        this->__action_start(start_action);

        this->lock.lock();

        // assert this->lock is locked
    }


public:
    ClusterStatusImpl(std::shared_ptr<RankManager> __rank_manager): running_actions(running_action_cmp), rank_manager(__rank_manager) { }
    ClusterStatusImpl(ClusterStatusImpl &&) = delete;
    ClusterStatusImpl(const ClusterStatusImpl &) = delete;
    ~ClusterStatusImpl() {this->master = nullptr;}

    virtual ComputeDomain
    GetMaxComputeDomain() override {
        return this->fake_compute_domains.size();
    }

    virtual Rank
    GetMaxRank() override {
         return this->accs.size();
    }

    virtual NBytes
    GetHBMUsage(ComputeDomain domain) override {
        return this->fake_compute_domains.at(domain)->used_hbm;
    }

    void
    Init_Stage_2(
        Timestamp now,
        const TransmitManager &transmit_manager,
        const GCG_Adding_OpManagement &GCG
    ) {
        this->now = now;
        for (auto [compute_domain, hbm_capability]: this->rank_manager->GetAllComputeDomains())
            this->fake_compute_domains.insert({compute_domain, std::make_shared<FakeComputeDomain>(hbm_capability)});

        for (Rank rank: this->rank_manager->GetAllRanks()) {
            ComputeDomain domain = this->rank_manager->GetComputeDomain(rank);
            this->accs.insert({rank, {this->fake_compute_domains.at(domain)}});
        }

        auto insert_action__to__list = [&] (std::shared_ptr<ActionForSimulation> action) {
            auto &acc = this->accs.at(action->rank);
            acc.__enqueue_action(action);
        };

        std::list<const Node *> done_nodes__with__undone__successor_or_transmit;

        // 1. Let's find all the non-ready or ready RunATenOps from un-done nodes
        for (auto &node: GCG) {
            if (node._.node_type != Node_For_Op)
                // Ignore schedule special nodes like Future
                continue;

            NodeID node_id = node.node_id;

            if (node._.done) {
                // "prim::Constant" nodes are included here

                bool has_undone_successor = false;
                for (auto _user: node.uses) {
                    auto *user = GCG.__get_node(_user);
                    if (!user->_.done) {
                        has_undone_successor = true;
                        break;
                    }
                }

                bool has_unpermitted_transmit = false;
                if (transmit_manager.node_id__to__transmit_ids.contains(node_id))
                    for (TransmitID transmit_id: transmit_manager.node_id__to__transmit_ids.at(node_id)) {
                        auto transmit_info = transmit_manager.transmit_id__to__info.at(transmit_id);
                        if (!transmit_info->has_been_permitted) {
                            has_unpermitted_transmit = true;
                            break;
                        }
                    }

                if (has_undone_successor || has_unpermitted_transmit)
                    done_nodes__with__undone__successor_or_transmit.push_back(&node);
            } else if (node._.is_scheduled) {
                assert(!node._.is_constant);
                Rank assigned_to = node._.assigned_to;
                IssuingID issuing_id = node._.issuing_id;

                for (auto input_id: node.inputs) {
                    if (GCG.checkpoint_prelocations.contains(input_id)) {
                        auto fetch_checkpoint = std::make_shared<ActionForSimulation>(
                            issuing_id, 9999, node._.execution_sequence, 0,
                            FetchCheckpoint, std::make_shared<AccActionParamPayload>(), std::vector<NodeID>(), input_id, std::nullopt,
                            1, true, Trace({input_id, std::nullopt}), false);
                        fetch_checkpoint->rank = assigned_to;
                        insert_action__to__list(fetch_checkpoint);
                    }

                }

                if (node._.has_tensor_payload
                    ||
                    (
                        // 这个flag指的是，该CKPT在另一个rank上Arrived，但是assigned_to上的report op done迟到，处于undone的状态
                        // 这个情形这么处理，我们先在这里加入upload_tensor算子，可以立刻完成并触发repor op done
                        GCG.checkpoint_locations.contains(node_id) && !GCG.checkpoint_locations.at(node_id).empty()
                    )
                ) {
                    auto param = std::make_shared<struct UploadTensor>();
                    param->node_id = node_id;
                    auto upload_tensor = std::make_shared<ActionForSimulation>(
                        issuing_id, 0, node._.execution_sequence, issuing_id,
                        UploadTensor, param, std::vector<NodeID>(), node_id, std::nullopt,
                        1, true, std::nullopt, false);
                    upload_tensor->rank = assigned_to;
                    insert_action__to__list(upload_tensor);  
                } else {
                    auto param = std::make_shared<struct RunATenOP>();
                    param->node_id = node_id;
                    param->task_id = node._.task_id;
                    param->task_node_id = node._.task_node_id;

                    auto action = std::make_shared<ActionForSimulation>(
                        issuing_id, 0, node._.execution_sequence, issuing_id,
                        RunATenOP, param, node.inputs, node_id, std::nullopt,
                        0, false, std::nullopt, false);
                    action->rank = assigned_to;
                    insert_action__to__list(action);
                }
            }
        }

        // 2. Let's settle all the checkpoint trace
        for (auto &e: GCG.checkpoint_prelocations) {
            auto ckpt_id = e.first;
            size_t node_size = total_size_of_vd(this->ask_node_shape(ckpt_id));
            auto *ckpt = GCG.__get_node(ckpt_id);
            for (Rank rank: e.second) {
                if (GCG.checkpoint_locations.at(ckpt_id).contains(rank)) {
                    // assert(ckpt->_.done);
                    // 上面这一句是我特地注释掉的
                    // 按理来说，CKPT在rank上，应该是done的
                    // 但可能因为report op done迟到，CKPT处于undone状态。
                    // 解法就是undone的算子发射一个upload_tensor算子立刻执行，并report_op_done
                    // preloc和loc，按照master中记录的ckpt位置情况，发射settled_as_checkpoint或者直接发射trace

                    auto &acc = this->accs.at(rank);
                    auto fake_payload = acc.malloc(node_size);;
                    this->__broadcast__to_non_ready_list(
                        rank, std::nullopt,
                        std::nullopt, nullptr,
                        Trace({ckpt_id, std::nullopt}), fake_payload);
                } else {
                    auto param = std::make_shared<struct SettledAsCheckpoint>();
                    param->node_id = ckpt_id;
                    auto action = std::make_shared<ActionForSimulation>(
                        -1/* 此时ckpt可能是undone、也可能未传输完毕；可能{ckpt_id}输入由Recv或RunATenOP进行broadcast */, 9999, 0, 0,
                        SettledAsCheckpoint, param, std::vector<NodeID>({ckpt_id}), std::nullopt, Trace({ckpt_id, std::nullopt}),
                        1, true, std::nullopt, false);
                    action->rank = rank;
                    insert_action__to__list(action);
                }
            }
        }

        // 3. Let's find all the transmit ops
        for (auto e: transmit_manager.transmit_id__to__info) {
            TransmitID transmit_id = e.first;
            auto transmit_info = e.second;
            std::optional<IssuingID> issuing_id = transmit_info->issuing_id;
            NodeID node_id = transmit_info->node__to_transmit;

            if (transmit_info->send_domain == transmit_info->recv_domain) {
                auto send = std::make_shared<ActionForSimulation>(
                    issuing_id, 9999, transmit_id, issuing_id.value(),
                    Transmit_Send, transmit_info, std::vector<NodeID>({node_id}), std::nullopt, std::nullopt,
                    1, true, std::nullopt, true);
                auto recv = std::make_shared<ActionForSimulation>(
                    issuing_id, 9998, transmit_id, issuing_id.value(),
                    Transmit_Recv, transmit_info, std::vector<NodeID>(), node_id, std::nullopt,
                    1, true, Trace({std::nullopt, transmit_id}), true);
                send->rank = transmit_info->send_rank;
                recv->rank = transmit_info->recv_rank;

                insert_action__to__list(send);
                insert_action__to__list(recv);
                continue;
            }

            // for transmit_info->send_domain != transmit_info->recv_domain case

            if (transmit_info->has_finished) {

                // 是不是要broadcast一下？
            } else if (transmit_info->has_been_permitted) {
            // } else if (transmit_info->has_requested_send == 1
            //     && transmit_info->has_been_permitted == 1
            //     && transmit_info->has_finished == 0) {
                auto send = std::make_shared<ActionForSimulation>(
                    issuing_id, 0, transmit_id, issuing_id.value(),
                    Transmit_Send, transmit_info, std::vector<NodeID>({node_id}), std::nullopt, std::nullopt,
                    0, true, Trace({std::nullopt, transmit_id}), true);
                // 虽然这对传输已经被允许了，但是仍然发射一个Transmit_AllocTensor算子，不want trace
                // 为了让HBM容量感知机制，有个Alloc Fake Payload的动作
                auto alloc_tensor = std::make_shared<ActionForSimulation>(
                    issuing_id, 9999, transmit_id, issuing_id.value(),
                    Transmit_AllocTensor, transmit_info, std::vector<NodeID>(), std::nullopt, Trace({std::nullopt, transmit_id + 1}),
                    1, false, std::nullopt, false);
                auto recv = std::make_shared<ActionForSimulation>(
                    issuing_id, 0, transmit_id, issuing_id.value(),
                    Transmit_Recv, transmit_info, std::vector<NodeID>(), node_id, std::nullopt,
                    0, true, Trace({std::nullopt, transmit_id + 1}), true);
                send->rank = transmit_info->send_rank;
                alloc_tensor->rank = transmit_info->recv_rank;
                recv->rank = transmit_info->recv_rank;

                this->send_actions[transmit_id] = send;

                insert_action__to__list(send);
                insert_action__to__list(alloc_tensor);
                insert_action__to__list(recv);
            } else if (transmit_info->has_requested_send) {
            // } else if (transmit_info->has_requested_send == 1
            //     && transmit_info->has_been_permitted == 0
            //     && transmit_info->has_finished == 0) {
                auto send = std::make_shared<ActionForSimulation>(
                    issuing_id, 0, transmit_id, issuing_id.value(),
                    Transmit_Send, transmit_info, std::vector<NodeID>({node_id}), std::nullopt, std::nullopt,
                    0, true, Trace({std::nullopt, transmit_id}), true);
                auto alloc_tensor = std::make_shared<ActionForSimulation>(
                    issuing_id, 0, transmit_id, issuing_id.value(),
                    Transmit_AllocTensor, transmit_info, std::vector<NodeID>(), std::nullopt, Trace({std::nullopt, transmit_id + 1}),
                    1, false, Trace({std::nullopt, transmit_id}), true);
                auto recv = std::make_shared<ActionForSimulation>(
                    issuing_id, 0, transmit_id, issuing_id.value(),
                    Transmit_Recv, transmit_info, std::vector<NodeID>(), node_id, std::nullopt,
                    0, true, Trace({std::nullopt, transmit_id + 1}), true);
                send->rank = transmit_info->send_rank;
                alloc_tensor->rank = transmit_info->recv_rank;
                recv->rank = transmit_info->recv_rank;

                this->send_actions[transmit_id] = send;

                insert_action__to__list(send);
                insert_action__to__list(alloc_tensor);
                insert_action__to__list(recv);
            } else {
            // } else if (transmit_info->has_requested_send == 0
            //     && transmit_info->has_been_permitted == 0
            //     && transmit_info->has_finished == 0) {
                auto request_send_to_master = std::make_shared<ActionForSimulation>(
                    issuing_id, 0, transmit_id, issuing_id.value(),
                    Transmit_RequestSendToMaster, transmit_info, std::vector<NodeID>({node_id}), std::nullopt, std::nullopt,
                    1, true, std::nullopt, false);
                auto send = std::make_shared<ActionForSimulation>(
                    issuing_id, 0, transmit_id, issuing_id.value(),
                    Transmit_Send, transmit_info, std::vector<NodeID>({node_id}), std::nullopt, std::nullopt,
                    0, true, Trace({std::nullopt, transmit_id}), true);
                auto alloc_tensor = std::make_shared<ActionForSimulation>(
                    issuing_id, 0, transmit_id, issuing_id.value(),
                    Transmit_AllocTensor, transmit_info, std::vector<NodeID>(), std::nullopt, Trace({std::nullopt, transmit_id + 1}),
                    1, false, Trace({std::nullopt, transmit_id}), true);
                auto recv = std::make_shared<ActionForSimulation>(
                    issuing_id, 0, transmit_id, issuing_id.value(),
                    Transmit_Recv, transmit_info, std::vector<NodeID>(), node_id, std::nullopt,
                    0, true, Trace({std::nullopt, transmit_id + 1}), true);
                request_send_to_master->rank = transmit_info->send_rank;
                send->rank = transmit_info->send_rank;
                alloc_tensor->rank = transmit_info->recv_rank;
                recv->rank = transmit_info->recv_rank;

                this->send_actions[transmit_id] = send;

                insert_action__to__list(request_send_to_master);
                insert_action__to__list(send);
                insert_action__to__list(alloc_tensor);
                insert_action__to__list(recv);
            }
        }

        // std::cout << "Simulation Constructing: Actions Pre-full" << std::endl;
        // this->debugOutput();
        // std::cout << "Simulation Constructing: Actions Pre-full Over" << std::endl;

        // 4. 已经将scheduled算子填进加速器的un-ready队列中了，
        //    对于那些前驱是done node的scheduled，相应的输入应该是ready，所以我们把done node影响到的所有rank都进行broadcast，
        //    把scheudled算子在加速器队列中的操作数搞成ready
        for (auto done_node: done_nodes__with__undone__successor_or_transmit) {
            auto done_node_id = done_node->node_id;
            size_t node_size = total_size_of_vd(this->ask_node_shape(done_node_id));

            if (done_node->_.is_constant) {
                // 常数本来就是随着RunATenOP直接发射出去的，所以就默认他们在所在机器上直接ready
                for (auto _user: done_node->uses) {
                    auto user = GCG.__get_node(_user);
                    if (!user->_.done) {
                        auto &acc = this->accs.at(user->_.assigned_to);
                        auto fake_payload = acc.malloc(node_size);
                        this->__broadcast__to_non_ready_list(
                            user->_.assigned_to, user->_.issuing_id,
                            done_node_id, fake_payload,
                            std::nullopt, nullptr);
                    }
                }
            } else {
                {
                    auto &acc = this->accs.at(done_node->_.assigned_to);
                    auto fake_payload = acc.malloc(node_size);
                    this->__broadcast__to_non_ready_list(
                        done_node->_.assigned_to, done_node->_.issuing_id,
                        done_node_id, fake_payload,
                        std::nullopt, nullptr);
                }

                if (transmit_manager.node_id__to__transmit_ids.contains(done_node_id))
                    for (auto transmit_id: transmit_manager.node_id__to__transmit_ids.at(done_node_id)) {
                        auto transmit_info = transmit_manager.transmit_id__to__info.at(transmit_id);
                        if (transmit_info->has_finished) {
                            auto &acc = this->accs.at(transmit_info->recv_rank);
                            auto fake_payload = acc.malloc(node_size);
                            this->__broadcast__to_non_ready_list(
                                transmit_info->recv_rank, done_node->_.issuing_id,
                                done_node_id, fake_payload,
                                std::nullopt, nullptr);
                        }
                    }
            }
        }
        
        // 5. 然后来处理一下checkpoint
        //    主要是针对那些，真实情况下，CKPT源头已经求解完毕，或者传输已经完成，SettledAsCheckpoint已被激活，
        //    但还没来得及向Master ReportCKPTArrived的SettledAsCheckpoint的情况
        //
        //    该真实情况下，在模拟器中，还认为这个checkpoint是未完成的，留下了SettledAsCheckpoint，但transmit已经finish
        //    所以，我们把这些checkpoint的原始求解位置和传输影响到的范围都broadcast一遍
        for (auto &e: GCG.checkpoint_prelocations) {
            auto ckpt_id = e.first;
            auto *ckpt_node = GCG.__get_node(ckpt_id);

            if (!ckpt_node->_.done)
                continue;

            size_t node_size = total_size_of_vd(this->ask_node_shape(ckpt_id));
            {
                auto &acc = this->accs.at(ckpt_node->_.assigned_to);
                auto fake_payload = acc.malloc(node_size);
                this->__broadcast__to_non_ready_list(
                    ckpt_node->_.assigned_to, std::nullopt,
                    ckpt_id, fake_payload,
                    std::nullopt, nullptr);
            }

            if (transmit_manager.node_id__to__transmit_ids.contains(ckpt_id))
                for (auto transmit_id: transmit_manager.node_id__to__transmit_ids.at(ckpt_id)) {
                    auto transmit_info = transmit_manager.transmit_id__to__info.at(transmit_id);
                    if (transmit_info->recv_for_settled_ckpt && transmit_info->has_finished) {
                        auto &acc = this->accs.at(transmit_info->recv_rank);
                        auto fake_payload = acc.malloc(node_size);
                        this->__broadcast__to_non_ready_list(
                            transmit_info->recv_rank, std::nullopt,
                            ckpt_id, fake_payload,
                            std::nullopt, nullptr);
                    }
                }
        }

        // 在这里，Scheduled算子、未完成的transmit、checkpoint都已经放在模拟器里了
        // 那些已经可以被执行的算子都成为了ready状态
        // 在模拟器构造最最最开始，ready算子放在running里面由Init_Stage_3完成
    }



    void
    Init_Stage_3(
        const TransmitManager &transmit_manager,
        const GCG_Adding_OpManagement &GCG) {
        Timestamp now = this->now;

        // 刚才的Init_Stage_2，Master 和 ClusterStatus 没连接起来，有ClusterSys数据结构

        // Init_Stage_3中需要Master 和 ClusterStatus连接起来
        // 因为SetteledAsCheckpoint向master汇报，然后Master可能撤回Cluster中的checkpoint


        // 6. 然后从这些ready状态的算子中，挑出来这一刻正在执行的算子，推测出来这些正在执行的算子是何时开始的
        //    并推测出来这些算子何时执行完毕。
        //    然后把模拟器中，全集群的running_actions给填上，设置正确的busy stream
        //
        //    接下来的逻辑只能自己手动搞，不能用复用整个集群的Step，因为我们现在就是在初始化running_actions的状态
        //    只能找出来每个加速器、每个stream队列中第一个正在执行的算子，然后放在running_actions
        //    而不是running_actions就开始流动了

        // a) SettledAsCheckpoint 和 FetchCheckpoint 的执行时间可忽略不计，而且能执行都直接执行
        //    而且Scheduled RunATenOP的input为checkpoint时，我们也没处理input，直接就用FetchCheckpoint
        //    所以干脆我们先把 ready list 里的 SettledAsCheckpoint 和 FetchCheckpoint 都执行、排除完，发挥他们的实际效力
        //    反正duration可忽略，重要的是那些真正需要花时间执行的算子，所以就这样了
        for (auto &p: this->accs) {  
            Rank rank = p.first;
            auto &acc = p.second;
            for (StreamID stream_id = 0; stream_id < NR_STREAM; stream_id++) {
                auto &ready_list = acc.streams[stream_id];
                while (!ready_list.empty()) {
                    auto action = ready_list.top();

                    if (action->acc_action_spec->op_enum != SettledAsCheckpoint
                        && action->acc_action_spec->op_enum != FetchCheckpoint
                        && action->acc_action_spec->op_enum != Transmit_AllocTensor)
                        // 因为刚才加入SettledAsCheckpoint/FetchCheckpoint/Transmit_AllocTensor的优先级都是9999
                        // 所以这里pop出来的都是这仨，先把这仨的效力都安排上
                        // 一旦没有这仨action了，就直接退出去
                        break;

                    ready_list.pop();
                    acc.__before_running_action(action, this->ask_node_shape);
                    this->__action_start(action);
                    acc.__broadcast__to_non_ready_list(
                        action->issuing_id,
                        action->acc_action_spec->output_id,
                        action->will_output,
                        action->acc_action_spec->trace_will_leave,
                        action->will_leave_as_trace);

                    this->__action_bottomhalf(action);
                }
            }
        }

        // b) 然后看看能执行的recv，send_domain != recv_domain的情况。
        //    为了执行这个recv，肯定有send跟这个recv配对
        //    这里我们不能直接执行__start_ready_action__for_idle_queue
        //    而是将__start_ready_action__for_idle_queue和__action_start直接写在这里
        //    因为这里我们就是想先处理recv（暂时先不管send和RunATenOP），因为正在进行中的recv意味着有一部分正在运行中的send，
        //    传输的优先级比计算的优先级高，所以我们先将正在运行的传输算子对给弄到running_actions里
        //    __start_ready_action__for_idle_queue可能还会处理send和RunATenOP，
        //
        //    在之后我们也不能直接用__start_ready_action__for_idle_queue，
        //    因为__start_ready_action__for_idle_queue会把算子的开始执行时间用this->now，直接入running_actions了
        //    但我们也要推测算子是什么时候开始执行的，比this->now更早 
        for (auto &p: this->accs) {
            Rank rank = p.first;
            auto &acc = p.second;
            for (StreamID stream_id = 0; stream_id < NR_STREAM; stream_id++) {
                auto &ready_list = acc.streams[stream_id];
                if (ready_list.empty())
                    continue;

                auto recv_action = ready_list.top();
                if (recv_action->acc_action_spec->op_enum != Transmit_Recv)
                    continue;

                ready_list.pop();

                acc.__before_running_action(recv_action, this->ask_node_shape);

                auto param = std::static_pointer_cast<struct TransmitInfo>(recv_action->acc_action_spec->op_param);

                // 既然Recv处于ready，那么这里处理的就不是send_domain == recv_domain的情况
                assert(param->send_domain != param->recv_domain);

                auto transmit_id = param->transmit_id;
                auto send_action = this->send_actions[transmit_id];
                this->send_actions.erase(transmit_id);

                recv_action->is_done_follower = 1;
                send_action->done_follower = recv_action;

#ifndef NDEBUG
                this->done_followers.insert(recv_action);
#endif

                acc.stream_busy[stream_id] = true;

                // 正在执行的recv，说明已经让对面的send处于ready状态了
                // 这里也等同于this->__action_start(recv_action)，但是不操纵整个集群的 running_actions
                this->__broadcast__to_non_ready_list(param->send_rank, recv_action->issuing_id,
                                                     std::nullopt, nullptr,
                                                     Trace({std::nullopt, param->transmit_id}), nullptr);
            }
        }

        // c) 然后处理可执行的recv and send，send_domain != recv_domain的情况。
        for (auto &p: this->accs) {
            Rank rank = p.first;
            auto &acc = p.second;
            for (StreamID stream_id = 0; stream_id < NR_STREAM; stream_id++) {
                auto &ready_list = acc.streams[stream_id];
                if (ready_list.empty())
                    continue;

                if (acc.stream_busy[stream_id])
                    continue;

                auto send_action = ready_list.top();
                if (send_action->acc_action_spec->op_enum != Transmit_Send)
                    continue;

                auto param = std::static_pointer_cast<struct TransmitInfo>(send_action->acc_action_spec->op_param);

                if (param->send_domain == param->recv_domain)
                    continue;

                // 接下来处理recv_domain != send_domain，recv 和 send 正在执行的情况

                ready_list.pop();

                acc.__before_running_action(send_action, this->ask_node_shape);

                send_action->start = now;
                // TODO: more exactly
                // 这里先随便写个send开始时间；这个send的开始执行是由recv触发的，要比permitted time晚一点
                // 因为permitted那瞬间，可能有RunATen正在执行
                // 在传输、计算互斥的情况下，我们设计send算子的执行优先级比RunATen要高，所以send的实际开始时间
                // 是permmitted time那一刻正在执行的RunATen的done timestamp

                auto transmit_id = param->transmit_id;
                send_action->predicted_end = send_action->start
                                           + this->transmit_predictor(param->send_rank,
                                                                      param->recv_rank,
                                                                      this->ask_node_shape(param->node__to_transmit));

                if (send_action->predicted_end < now)
                    // for profiling error
                    send_action->predicted_end = now;


                assert(send_action->done_follower);
                send_action->done_follower->predicted_end = send_action->predicted_end;
                this->running_actions.push(send_action->done_follower);
#ifndef NDEBUG
                this->done_followers.erase(send_action->done_follower);
#endif

                this->running_actions.push(send_action);
                acc.stream_busy[stream_id] = true;
                // 这里的动作应该和this->__action_start(send_action)一样，但send_action实际上没有动作，所以就无了
            }
        }




        // d) 然后处理正在执行的send，send_domain == recv_domain的情况
        for (size_t i = 0; i < 2; i++) {
            for (auto &p: this->accs) {
                Rank rank = p.first;
                auto &acc = p.second;
                for (StreamID stream_id = 0; stream_id < NR_STREAM; stream_id++) {
                    auto &ready_list = acc.streams[stream_id];

                    if (acc.stream_busy[stream_id])
                        continue;

                    while (!ready_list.empty()) {
                        auto action = ready_list.top();

                        if ((i == 0 && action->acc_action_spec->op_enum == Transmit_Send)
                            || (i == 1 && action->acc_action_spec->op_enum == Transmit_Recv)) {
                        } else
                            break;

                        auto param = std::static_pointer_cast<struct TransmitInfo>(action->acc_action_spec->op_param);

                        assert(param->send_domain == param->recv_domain);

                        // 第一次迭代处理send，第二次迭代处理由send激活的recv

                        ready_list.pop();
                        acc.__before_running_action(action, this->ask_node_shape);
                        this->__action_start(action);
                        acc.__broadcast__to_non_ready_list(
                            action->issuing_id,
                            action->acc_action_spec->output_id,
                            action->will_output,
                            action->acc_action_spec->trace_will_leave,
                            action->will_leave_as_trace);

                        this->__action_bottomhalf(action);
                    }
                }
            }
        }


        // e) 然后处理正在执行的RunATen
        for (auto &p: this->accs) {
            Rank rank = p.first;
            auto &acc = p.second;
            for (StreamID stream_id = 0; stream_id < NR_STREAM; stream_id++) {
                auto &ready_list = acc.streams[stream_id];
                if (ready_list.empty())
                    continue;

                if (acc.stream_busy[stream_id])
                    continue;

                auto action = ready_list.top();
                if (action->acc_action_spec->op_enum != RunATenOP)
                    continue;

                ready_list.pop();
                
                acc.__before_running_action(action, this->ask_node_shape);

                NodeID node_id = action->acc_action_spec->output_id.value();
                auto *node = GCG.__get_node(node_id);

                // 一个算子的开始执行时间是，max of (所有input的ready时间、issuing timestamp、上个执行算子的done timestamp)
                // 而且，我们假设mark node done的顺序和实际算子执行顺序相同，
                // 既然算子还没有收到mark node done，那么说明正在执行的算子一定在最后一个mark node done之后
                auto [_, last_done_timestamp] = GCG.last_done_node__for_each_rank.at(rank);
                Timestamp start = std::max(node->_.issuing_timestamp, last_done_timestamp);
                for (auto input_id: action->acc_action_spec->input_ids) {
                    auto *input = GCG.__get_node(input_id);
                    if (input->_.is_constant) {
                    } else if (input->_.assigned_to == rank)
                        // input在本地，不管是不是checkpoint都这么做
                        start = std::max(start, input->_.done_timestamp);
                    else {
                        // input在远程，不管是不是checkpoint都这么做
                        if (transmit_manager.recv__to__transmit_id.contains({input_id, rank})) {
                            TransmitID transmit_id = transmit_manager.recv__to__transmit_id.at({input_id, rank});
                            auto transmit_info = transmit_manager.transmit_id__to__info.at(transmit_id);
                            if (transmit_info->has_finished)
                                start = std::max(start, transmit_info->finish_time);
                        }
                    }
                }
                action->start = start;
                if (action->start)
                    // 开始时间估计错误
                    action->start = now;

                auto param = std::static_pointer_cast<struct RunATenOP>(action->acc_action_spec->op_param);
                action->predicted_end = action->start
                                      + this->task_predictor(this->rank_manager->GetModelByHost(this->rank_manager->GetHostByRank(rank)),
                                                             param->task_id,
                                                             param->task_node_id);
                if (action->predicted_end < now)
                    // for profiling error
                    action->predicted_end = now;

                this->running_actions.push(action);
                acc.stream_busy[stream_id] = true;
                // fake_cluster->__action_start(send_action);
            }
        }


        // e) 反正其他类型的action的duration为0，所以就直接__start_ready_action__for_idle_queue了
        this->lock.lock();
        for (auto &p: this->accs) {
            Rank rank = p.first;
            for (StreamID stream_id = 0; stream_id < NR_STREAM; stream_id++)
                this->__start_ready_action__for_idle_queue(rank, stream_id);
        }
        this->lock.unlock();

#ifndef NDEBUG
        this->check();
#endif   
    }

    virtual void
    SetMaster(std::shared_ptr<Master> p) override {
        this->master = p;
        this->Cluster::SetMaster(p);
    }

    virtual void
    debugOutput() override {
        std::cout << "Simulator internal ===================================================" << std::endl;
        {
            std::cout << "Running actions:" << std::endl;
            for (auto action: this->running_actions.heap)
                std::cout << *action << std::endl;

#ifndef NDEBUG
            std::cout << "Running actions (done followers):" << std::endl;
            for (auto action: this->done_followers)
                std::cout << *action << std::endl;
#endif
        }

        for (Rank rank: this->rank_manager->GetAllRanks()) {
            auto &acc = this->accs.at(rank);

            for (StreamID stream_id = 0; stream_id < NR_STREAM; stream_id++) {
                std::cout << "Rank[" << rank << "] stream[" << stream_id << "] busy: " << acc.stream_busy[stream_id] << std::endl;
            }

            std::cout << "Rank[" << rank << "] ready list:" << std::endl;
            for (StreamID stream_id = 0; stream_id < NR_STREAM; stream_id++) {
                for (auto action: acc.streams[stream_id].heap)
                    std::cout << *action << std::endl;
            }

            std::cout << "Rank[" << rank << "] non-ready list:" << std::endl;
            for (auto &p: acc.queues) {
                auto &non_ready_list = p.second;
                for (auto action: non_ready_list)
                    std::cout << *action << std::endl;
            }

            std::cout << "Rank[" << rank << "] Trace:" << std::endl;
            nlohmann::json trace_j = acc.traces_in_queue;
            std::cout << trace_j.dump() << std::endl;
            
            std::cout << "Rank[" << rank << "] Checkpoint:" << std::endl;
            nlohmann::json ckpt_j = acc.trace_payloads;
            std::cout << ckpt_j.dump() << std::endl;
        }
        std::cout << "===================================================" << std::endl;
    }

};


#endif