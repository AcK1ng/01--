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


struct semaphore_wrapper {
    //https://stackoverflow.com/questions/73449114/how-to-declare-and-initialize-a-vector-of-semaphores-in-c
    std::counting_semaphore<SEM_VALUE_MAX> sem;
    semaphore_wrapper(): sem(0) {}
    void acquire() {this->sem.acquire();}
    void release() {this->sem.release();}
};

class SimpleOOQueueImpl;

class SimpleOOQueueThread {
public:
    SimpleOOQueueThread(SimpleOOQueueImpl *queue): queue(queue) {}
    void operator()(StreamID stream_id);
private:
    SimpleOOQueueImpl *queue;
};

class SimpleOOQueueImpl: public OOQueue {
    friend class SimpleOOQueueThread;
public:
    SimpleOOQueueImpl(StreamID nr_stream) {
        this->nr_stream = nr_stream;
        this->ready_lists = std::vector<std::priority_queue<Action>>(nr_stream);
        this->ready_list_locks = std::vector<std::mutex>(nr_stream);
        this->ready_list_exec_locks = std::vector<std::mutex>(nr_stream);
        this->ready_nr_semaphores = std::vector<struct semaphore_wrapper>(nr_stream);
    }

    virtual void StartUp() override {
        for (StreamID stream_id = 0; stream_id < this->nr_stream; stream_id++) {
            std::thread t(SimpleOOQueueThread(this), stream_id);
            t.detach();
        }

        this->OOQueue::StartUp();
    }

    virtual void LeaveTraceFromRemote(Trace trace,
            std::shared_ptr<TracePayload> payload,
            std::optional<IssuingID> hint_issuing_id) override {
        this->nonready_list_lock.lock();
        this->__leave_trace(trace, payload);
        this->nonready_list_lock.unlock();
    }
    virtual void EnqueueActions(std::optional<IssuingID>, std::vector<std::shared_ptr<AccActionSpec>> &) override;
    virtual void PurgeTrace(Trace trace) override;
private:
    void __push_to_ready_list(Action &op);

    void __broadcast_operand(RegID id,
            std::shared_ptr<RegPayload> payload);
    void __leave_trace(Trace trace,
            std::shared_ptr<TracePayload> payload); // for who provides trace
    std::pair<bool, std::shared_ptr<TracePayload>> __fetch_trace__no_lock(Trace trace,
            bool clear_trace); // for who needs trace

    StreamID nr_stream;

    std::vector<std::mutex> ready_list_locks;
    std::vector<std::mutex> ready_list_exec_locks;
    std::vector<std::priority_queue<Action>> ready_lists;
    std::vector<struct semaphore_wrapper> ready_nr_semaphores;

    std::mutex nonready_list_lock;
    std::multiset<Trace> traces_will_be_left;
    std::multiset<Trace> traces_will_be_deleted;
    std::multimap<Trace, std::shared_ptr<TracePayload>> traces;
    std::list<Action> nonready_list;
};

void
SimpleOOQueueImpl::EnqueueActions(std::optional<IssuingID> issuing_id,
                                  std::vector<std::shared_ptr<AccActionSpec>> &op_specs) {
    this->nonready_list_lock.lock();
    for (auto it: op_specs) {
        auto [op_tophalf, op_bottomhalf] = this->GetActionFun(it->op_enum);
        Action op(issuing_id, *it, op_tophalf, op_bottomhalf);
        if (op.acc_action_spec.wanted_trace.has_value()) {
            auto trace = op.acc_action_spec.wanted_trace.value();
            auto [trace_gotcha, trace_payload] =
                this->__fetch_trace__no_lock(trace,
                                             op.acc_action_spec.clear_trace);
            if (trace_gotcha)
                op.provide_trace(trace, trace_payload);
        }
        if (op.acc_action_spec.trace_will_leave.has_value())
            this->traces_will_be_left.insert(op.acc_action_spec.trace_will_leave.value());
        if (op.ready()) {
            this->__push_to_ready_list(op);
        } else {
            this->nonready_list.push_back(op);
        }
    }
    this->nonready_list_lock.unlock();
}

void
SimpleOOQueueImpl::PurgeTrace(Trace trace) {
    this->nonready_list_lock.lock();
    this->traces.erase(trace);
    size_t nr_traces_will_gen = this->traces_will_be_left.count(trace);
    size_t nr_traces_will_be_deleted = this->traces_will_be_deleted.count(trace);
    for (size_t i = 0; i < nr_traces_will_gen - nr_traces_will_be_deleted; i++)
        this->traces_will_be_deleted.insert(trace);
    this->nonready_list_lock.unlock();
}

void
SimpleOOQueueImpl::__push_to_ready_list(Action &op) {
    StreamID stream_id = op.acc_action_spec.stream_id;
    auto &ready_list = this->ready_lists[stream_id];
    auto &ready_list_lock = this->ready_list_locks[stream_id];
    auto &ready_nr_semaphore = this->ready_nr_semaphores[stream_id];

    assert(stream_id < this->nr_stream);

#ifdef DEBUG_OOQUEUE
    // std::cout << "stream[" << stream_id << "] ready list lock" << std::endl;
#endif
    ready_list_lock.lock();
    ready_list.push(op);

    ready_nr_semaphore.release();

    ready_list_lock.unlock();
#ifdef DEBUG_OOQUEUE
    // std::cout << "stream[" << stream_id << "] ready list unlock" << std::endl;
#endif
}

void
SimpleOOQueueImpl::__broadcast_operand(RegID id, std::shared_ptr<RegPayload> payload) {
    this->nonready_list_lock.lock();
    for (auto it = this->nonready_list.begin(); it != this->nonready_list.end(); ) {
        it->provide_input(id, payload);
        if (it->ready()) {
            this->__push_to_ready_list(*it);
            it = this->nonready_list.erase(it);
        } else {
            it++;
        }
    }
    this->nonready_list_lock.unlock();
}

void
SimpleOOQueueImpl::__leave_trace(Trace trace, std::shared_ptr<TracePayload> payload) {
    // need nonready list lock
    do {
        if (auto it = this->traces_will_be_deleted.find(trace);
            it != this->traces_will_be_deleted.end()) {
            this->traces_will_be_deleted.erase(it);
            break;
        }

        bool the_trace_has_been_cleared = false;
        for (auto it = this->nonready_list.begin(); it != this->nonready_list.end(); ) {
            the_trace_has_been_cleared = it->provide_trace(trace, payload);
            if (it->ready()) {
                this->__push_to_ready_list(*it);
                it = this->nonready_list.erase(it);
            } else {
                it++;
            }
            if (the_trace_has_been_cleared)
                break;
        }

        if (!the_trace_has_been_cleared) {
            // nobody needs the trace
            this->traces.insert({trace, payload});
        }
    } while(0);
}

std::pair<bool, std::shared_ptr<TracePayload>>
SimpleOOQueueImpl::__fetch_trace__no_lock(Trace trace,
        bool clear_trace) {
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
}

void
SimpleOOQueueThread::operator()(StreamID stream_id) {
    auto queue = this->queue;
    auto &ready_list = queue->ready_lists[stream_id];
    auto &ready_list_lock = queue->ready_list_locks[stream_id];
    auto &exec_lock = queue->ready_list_exec_locks[stream_id];
    auto &ready_nr_semaphore = queue->ready_nr_semaphores[stream_id];
    auto acc = queue->acc;
    acc->init_for_each_stream_thread();

    while (1) {
        ready_nr_semaphore.acquire();

        exec_lock.lock();

#ifdef DEBUG_OOQUEUE
    // std::cout << "stream[" << stream_id << "] ready list lock" << std::endl;
#endif
        ready_list_lock.lock();
        auto op = ready_list.top();
        ready_list.pop();
        ready_list_lock.unlock();
#ifdef DEBUG_OOQUEUE
//    std::cout << "stream[" << stream_id << "] ready list unlock" << std::endl;
#endif

        try {

#ifdef DEBUG_OOQUEUE
            std::ostringstream os1;
            os1 << "Running Action[" << RealTimeNow() << "]: " << op << std::endl;
            std::cout << os1.str();
#endif

            assert(op.ready());
            auto [output, info_to_bottomhalf, trace_payload] = op.op_tophalf(op.inputs,
                                                                             op.trace_payload,
                                                                             op.acc_action_spec.op_param,
                                                                             op.issuing_id,
                                                                             acc);
#ifdef DEBUG_OOQUEUE
            std::ostringstream os;
            os << "Tophalf OK[" << RealTimeNow() << "]: " << std::endl;
            std::cout << os.str();
#endif
            op.op_bottomhalf(info_to_bottomhalf, op.issuing_id, acc);

            exec_lock.unlock();

#ifdef DEBUG_OOQUEUE
            std::ostringstream os2;
            os2 << "Running Done[" << RealTimeNow() << "]!";
            std::cout << os2.str() << std::endl;
#endif

            if (op.acc_action_spec.output_id.has_value())
                queue->__broadcast_operand(op.acc_action_spec.output_id.value(),output);
            if (op.acc_action_spec.trace_will_leave.has_value()) {
                Trace trace = op.acc_action_spec.trace_will_leave.value();
                queue->nonready_list_lock.lock();
                if (auto it = queue->traces_will_be_left.find(trace);
                    it != queue->traces_will_be_left.end())
                    queue->traces_will_be_left.erase(it);
                queue->__leave_trace(trace, trace_payload);
                queue->nonready_list_lock.unlock();
            }

        } catch (std::exception e) {
            assert(0);
            std::cout << "op exec failed: " << op << std::endl;
            exit(-1);
        }
    }
}


std::unique_ptr<class OOQueue>
GetSimpleOOQueue(StreamID nr_stream) {
    return std::make_unique<SimpleOOQueueImpl>(nr_stream);
}
