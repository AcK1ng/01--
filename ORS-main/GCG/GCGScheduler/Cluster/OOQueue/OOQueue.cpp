#include "OOQueue.h"

#include <map>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include <cstddef> // for size_t
#include <cassert>
#include <thread>
#include <queue>
#include <list>
#include <map>
#include <semaphore>
#include <mutex>
#include <utility>
#include <iostream>

#include <hwloc.h>

hwloc_topology_t topology;

struct semaphore_wrapper {
    //https://stackoverflow.com/questions/73449114/how-to-declare-and-initialize-a-vector-of-semaphores-in-c
    std::counting_semaphore<SEM_VALUE_MAX> sem;
    semaphore_wrapper(): sem(0) {}
    void acquire() {this->sem.acquire();}
    void release() {this->sem.release();}
};

class BottomHalf_Listener {
public:
    BottomHalf_Listener(Accelerator *acc, std::pair<CoreID, LCoreID> bottomhalf_used_core): 
        acc(acc), bottomhalf_used_core(bottomhalf_used_core) { }

    void operator()() {
        auto [core_id, lcore_id] = this->bottomhalf_used_core;
        hwloc_obj_t core = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, core_id);
        hwloc_obj_t logical_processor = core->children[lcore_id];
        std::cout << "Rank[" << acc->GetRank() << "] bottomhalf thread place on " << "core_id: " << core_id << " lcore_id: " << lcore_id << std::endl;
        hwloc_set_cpubind(topology, logical_processor->cpuset, HWLOC_CPUBIND_THREAD);

        while (true) {
            this->bottom_half_reqs.acquire();

            this->lock.lock();
            auto &[op_bottomhalf, info_to_bottomhalf, issuing_id] = this->reqs.front();
            this->lock.unlock();

            op_bottomhalf(info_to_bottomhalf, issuing_id, this->acc);
            
            std::lock_guard<std::mutex> holding_lock(this->lock);
            this->reqs.pop_front();
        }
    }

    inline void
    append(OOQueueBottomHalfActionFuncType op_bottomhalf,
           std::shared_ptr<InfoToBottomHalf> info_to_bottomhalf,
           std::optional<IssuingID> issuing_id) {
        {
            std::lock_guard<std::mutex> holding_lock(this->lock);
            this->reqs.push_back({op_bottomhalf, info_to_bottomhalf, issuing_id});
        }
        this->bottom_half_reqs.release();
    }

private:
    Accelerator *acc;
    struct semaphore_wrapper bottom_half_reqs;
    std::list<std::tuple<OOQueueBottomHalfActionFuncType,
                         std::shared_ptr<InfoToBottomHalf>,
                         std::optional<IssuingID>>> reqs;
    std::pair<CoreID, LCoreID> bottomhalf_used_core;
    std::mutex lock;
};

inline static bool
operator<(const std::shared_ptr<Action> &op1, const std::shared_ptr<Action> &op2) {
    if (op1->inner_priority != op2->inner_priority)
        return op1->inner_priority < op2->inner_priority;
    return op1->acc_action_spec.acc_action_base < op2->acc_action_spec.acc_action_base;
}

class OOQueueImpl: public OOQueue {
public:
    OOQueueImpl(StreamID nr_stream,
                std::vector<std::unordered_set<std::tuple<CoreID, LCoreID>>> top_half_used_cores,
                std::pair<CoreID, LCoreID> bottomhalf_used_core):
        top_half_used_cores(top_half_used_cores),
        bottomhalf_used_core(bottomhalf_used_core),
        nr_stream(nr_stream),
        ready_list_locks(nr_stream),
        ready_list_exec_locks(nr_stream),
        ready_nr_semaphores(nr_stream) {
        for (StreamID stream_id = 0; stream_id < nr_stream; stream_id++)
            this->ready_lists.emplace_back(my_priority_queue<std::shared_ptr<Action>>(
                [](const std::shared_ptr<Action>& l, const std::shared_ptr<Action>& r) {return *(l.get()) < *(r.get());}
                )
            );
        }

    void operator()(StreamID stream_id) {
        hwloc_bitmap_t bitmap = hwloc_bitmap_alloc();
        hwloc_bitmap_t t = hwloc_bitmap_alloc();
        for (auto [core_id, lcore_id]: this->top_half_used_cores[stream_id]) {
            hwloc_obj_t core = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, core_id);
            hwloc_obj_t logical_processor = core->children[lcore_id];
            std::cout << "Rank[" << this->acc->GetRank() << "] " << "StreamID[" << stream_id << "] place on " << "core_id: " << core_id << " lcore_id: " << lcore_id << std::endl;
            hwloc_bitmap_or(bitmap, t, logical_processor->cpuset);
            hwloc_bitmap_copy(t, bitmap);
        }
        hwloc_set_cpubind(topology, bitmap, HWLOC_CPUBIND_THREAD);
        hwloc_bitmap_free(bitmap);
        hwloc_bitmap_free(t);

        auto &ready_list = this->ready_lists[stream_id];
        auto &ready_list_lock = this->ready_list_locks[stream_id];
        auto &exec_lock = this->ready_list_exec_locks[stream_id];
        auto &ready_nr_semaphore = this->ready_nr_semaphores[stream_id];
        auto acc = this->acc;
        acc->init_for_each_stream_thread();
        while (1) {
            ready_nr_semaphore.acquire();

            exec_lock.lock();
            ready_list_lock.lock();
            auto op = ready_list.top();
            ready_list.pop();

#ifdef DEBUG_OOQUEUE
            std::cout << "stream[" << stream_id << "] " << "from ready list out " << op->acc_action_spec.acc_action_base << std::endl;
#endif

            ready_list_lock.unlock();
            // try {
                auto [output, info_to_bottomhalf, trace_payload] = op->op_tophalf(op->inputs,
                                                                                  op->trace_payload,
                                                                                  op->acc_action_spec.op_param,
                                                                                  op->issuing_id,
                                                                                  acc);
                exec_lock.unlock();
                this->bottom_half_listener->append(op->op_bottomhalf, info_to_bottomhalf, op->issuing_id);

                std::lock_guard<std::mutex> holding_lock(this->nonready_list_lock);
                __broadcast_opr__and__trace(op->issuing_id,
                                            op->acc_action_spec.output_id, output,
                                            op->acc_action_spec.trace_will_leave, trace_payload);
                if (op->acc_action_spec.trace_will_leave.has_value()) {
                    Trace trace = op->acc_action_spec.trace_will_leave.value();
                    if (auto it = this->traces_will_be_left.find(trace);
                        it != this->traces_will_be_left.end())
                        this->traces_will_be_left.erase(it);
                }
            // } catch (std::exception e) {
            //     assert(0);
            //     std::cout << "op exec failed: " << op << std::endl;
            //     exit(-1);
            // }
        }
    }

    virtual void StartUp() override {
        {
            this->bottom_half_listener = std::make_unique<BottomHalf_Listener>(this->acc,
                                                                               this->bottomhalf_used_core);
            std::thread t([&]() {(*(this->bottom_half_listener))();});
            t.detach();
        }

        for (StreamID stream_id = 0; stream_id < this->nr_stream; stream_id++) {
            std::thread t([this, stream_id]() {(*(this))(stream_id);});
            t.detach();
        }
        
        this->OOQueue::StartUp();
    }

    
    virtual void
    LeaveTraceFromRemote(Trace trace,
                         std::shared_ptr<TracePayload> payload,
                         std::optional<IssuingID> hint_issuing_id) override {
        std::lock_guard<std::mutex> holding_lock(this->nonready_list_lock);
        this->__broadcast_opr__and__trace(hint_issuing_id, std::nullopt, nullptr, trace, payload);
    }

    virtual void
    EnqueueActions(std::optional<IssuingID> issuing_id,
                   std::vector<std::shared_ptr<AccActionSpec>> &op_specs) override {
        if (op_specs.size() == 0)
            return;

        std::lock_guard<std::mutex> holding_lock1(this->nonready_list_lock);
        if (!this->nonready_lists.contains(issuing_id)) {
            this->nonready_lists[issuing_id];
        }
        auto &nonready_list = this->nonready_lists.at(issuing_id);
        auto __fetch_trace__no_lock = [&] (Trace trace, bool clear_trace) -> std::pair<bool, std::shared_ptr<TracePayload>> {
            bool fetch_trace = false;
            std::shared_ptr<TracePayload> payload;

            auto it = this->traces.find(trace);
            if (it != this->traces.end()) {
                fetch_trace = true;
                payload = it->second;
                if (clear_trace)
                    this->traces.erase(it);
            }

            return {fetch_trace, payload};
        };
        for (auto &it: op_specs) {
            auto [op_tophalf, op_bottomhalf] = this->GetActionFun(it->op_enum);
            auto op = std::make_shared<Action>(issuing_id, *it, op_tophalf, op_bottomhalf);
            if (op->acc_action_spec.trace_will_leave.has_value())
                this->traces_will_be_left.insert(op->acc_action_spec.trace_will_leave.value());
            if (op->acc_action_spec.wanted_trace.has_value()) {
                auto trace = op->acc_action_spec.wanted_trace.value();
                auto [trace_gotcha, trace_payload] =
                    __fetch_trace__no_lock(trace, op->acc_action_spec.clear_trace);
                if (trace_gotcha)
                    op->provide_trace(trace, trace_payload);
                else
                    this->who_need_trace[trace].insert(issuing_id);
            }
            if (op->ready())
                this->__push_to_ready_list(op);
            else
                nonready_list.push_back(op);
        }
        if (nonready_list.empty())
            this->nonready_lists.erase(issuing_id);
    }

    virtual void
    PurgeTrace(Trace trace) override {
        std::lock_guard<std::mutex> holding_lock(this->nonready_list_lock);
        this->traces.erase(trace);
        size_t nr_traces_will_gen = this->traces_will_be_left.count(trace);
        size_t nr_traces_will_be_deleted = this->traces_will_be_deleted.count(trace);
        // 这里大部分情况应该都是 nr_traces_will_be_deleted == 0；
        // 这里是考虑到多次PurgeTrace；多批次发射留下相同trace算子，穿插的情况
        for (size_t i = 0; i < nr_traces_will_gen - nr_traces_will_be_deleted; i++)
            this->traces_will_be_deleted.insert(trace);
    }

private:
    // 这里的接口设计原则，整个队列的动作如下
    // 1. （本地计算或者LeaveTrace） 发Trace+作用于non ready list所有op
    // 2. （EnqueueActions）找Trace+作用于non ready list中单个op
    // 3. （本地计算RegID），作用域non ready list所有op
    // 4. 单个op变成ready，送进ready list
    //
    // 1和3合成__broadcast_opr__and__trace
    // 2在EnqueueActions里
    // 4是__push_to_ready_list

    inline void
    __broadcast_opr__and__trace(std::optional<IssuingID> issuing_id,
                                std::optional<RegID> reg_id, std::shared_ptr<RegPayload> reg_payload,
                                std::optional<Trace> trace, std::shared_ptr<TracePayload> trace_payload) {
        std::unordered_set<std::optional<IssuingID>> non_ready_lists__to__broadcast = {-1, std::nullopt};
        non_ready_lists__to__broadcast.insert(issuing_id);

        if (reg_id.has_value()) {
            for (auto &issuing_id: non_ready_lists__to__broadcast) {
                if (!this->nonready_lists.contains(issuing_id))
                    continue;

                auto &nonready_list = this->nonready_lists.at(issuing_id);
                for (auto it = nonready_list.begin(); it != nonready_list.end(); ) {
                    (*it)->provide_input(reg_id.value(), reg_payload);
                    if ((*it)->ready()) {
                        this->__push_to_ready_list(*it);
                        it = nonready_list.erase(it);
                    } else
                        it++;
                }
            }
        }
        
        do {
            if (!trace.has_value())
                break;

            bool trace_gone = false;
            if (this->who_need_trace.contains(trace.value())) {
                for (auto &issuing_id: this->who_need_trace.at(trace.value())) {
                    if (!this->nonready_lists.contains(issuing_id))
                        continue;

                    non_ready_lists__to__broadcast.insert(issuing_id);

                    auto &nonready_list = this->nonready_lists.at(issuing_id);
                    for (auto it = nonready_list.begin(); it != nonready_list.end(); ) {
                        trace_gone = (*it)->provide_trace(trace.value(), trace_payload);
                        if ((*it)->ready()) {
                            this->__push_to_ready_list(*it);
                            it = nonready_list.erase(it);
                        } else
                            it++;
                        if (trace_gone)
                            break;
                    }
                    if (trace_gone)
                        break;
                }
                this->who_need_trace.erase(trace.value());
            }

            if (auto it = this->traces_will_be_deleted.find(trace.value());
                it != this->traces_will_be_deleted.end()) {
                this->traces_will_be_deleted.erase(it);
                break;
            }

            if (!trace_gone) {
                this->traces.insert({trace.value(), trace_payload});
                break;
            }

        } while(0);

        for (auto &issuing_id: non_ready_lists__to__broadcast)
            if (this->nonready_lists.contains(issuing_id) && this->nonready_lists.at(issuing_id).empty())
                this->nonready_lists.erase(issuing_id);
    }

    inline void
    __push_to_ready_list(std::shared_ptr<Action> &op) {
        StreamID stream_id = op->acc_action_spec.stream_id;
        auto &ready_list = this->ready_lists[stream_id];
        auto &ready_list_lock = this->ready_list_locks[stream_id];
        auto &ready_nr_semaphore = this->ready_nr_semaphores[stream_id];

        assert(stream_id < this->nr_stream);

        std::lock_guard<std::mutex> holding_lock(ready_list_lock);
#ifdef DEBUG_OOQUEUE
        std::cout << "stream[" << stream_id << "] " << "enqueue ready list " << op->acc_action_spec.acc_action_base << std::endl;
#endif
        ready_list.push(op);
        ready_nr_semaphore.release();
    }

    std::vector<std::unordered_set<std::tuple<CoreID, LCoreID>>> top_half_used_cores;
    std::pair<CoreID, LCoreID> bottomhalf_used_core;
    std::shared_ptr<BottomHalf_Listener> bottom_half_listener;

    StreamID nr_stream;

    // 这里拿锁顺序，先拿nonready_list_lock，再拿ready_list_lock
    std::vector<std::mutex> ready_list_locks;
    std::vector<std::mutex> ready_list_exec_locks;
    std::vector<my_priority_queue<std::shared_ptr<Action>>> ready_lists;
    std::vector<struct semaphore_wrapper> ready_nr_semaphores;

    // 在PurgeTrace的时候，已经留下的trace当然要删除
    // 关键就是放进ready_list里的算子，将会留下的trace不好处理
    // 所以在这里打个补丁，记录下来将会left的所有Trace
    // 当PurgeTrace的时候，我们在traces_will_be_deleted中做个标记
    // 当算子执行完成后，如果这个算子留下的trace在traces_will_be_deleted中，那就不要留下
    
    std::mutex nonready_list_lock;

    // 指的是在non ready list和ready list中，会留下trace的算子数量
    std::multiset<Trace> traces_will_be_left;
    // 指的是，从ready list中结束的算子留下的trace，有多少trace将会删掉；当算子执行完毕后，留下的trace立刻删除
    std::multiset<Trace> traces_will_be_deleted;
    std::multimap<Trace, std::shared_ptr<TracePayload>> traces;
    std::unordered_map<Trace, std::unordered_set<std::optional<IssuingID>>> who_need_trace;

    std::unordered_map<std::optional<IssuingID>, std::list<std::shared_ptr<Action>>> nonready_lists;

};


std::unique_ptr<class OOQueue>
GetOOQueue(StreamID nr_stream,
           std::vector<std::unordered_set<std::tuple<CoreID, LCoreID>>> top_half_used_cores,
           std::pair<CoreID, LCoreID> bottomhalf_used_core) {
    return std::make_unique<OOQueueImpl>(nr_stream, top_half_used_cores, bottomhalf_used_core);
}
