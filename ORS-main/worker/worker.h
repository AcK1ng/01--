#pragma once
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include <atomic>
#include "master/operator.h"
#include "common/type.h"
#include "master/rpool.h"
#include <mutex>
#include <vector>

namespace ors {

class OpQueue {
public:
    OpQueue() {}
    ~OpQueue() {}

    void push(scoped_refptr<Operator>& op) {
        std::unique_lock<std::mutex> lck(_mutex);
        _queue.push_back(op);
    }

    scoped_refptr<Operator> pop() {
        std::unique_lock<std::mutex> lck(_mutex);
        if (_queue.empty()) {
            return nullptr;
        }
        auto op = _queue.front();
        _queue.erase(_queue.begin());
        return op;
    }

    void swap(std::vector<scoped_refptr<Operator>>& other) {
        std::unique_lock<std::mutex> lck(_mutex);
        _queue.swap(other);
    }

    size_t size() {
        std::unique_lock<std::mutex> lck(_mutex);
        return _queue.size();
    }
private:
    std::vector<scoped_refptr<Operator>> _queue;
    std::mutex _mutex;
};

class Worker {
public:
    Worker(uint32_t worker_id,
           c10::DeviceType dev_type,
           c10::DeviceIndex dev_index = -1);
    ~Worker() {}

    int start();
    void stop();

    void enqueue_compute(scoped_refptr<Operator>& op);
    void enqueue_transfer(scoped_refptr<Operator>& op);

    int32_t get_device_id() {
        return _device.index();
    }
    
    int get_worker_id() {
        return _worker_id;
    }

    static void CUDART_CB on_transfer_callback(void* data);
private:
    void sync_transfer_dependency_recursive(
        Operator* parent, 
        Operator* current_op, 
        std::unordered_set<Operator*>& visited
    );
    void sync_transfer_dependencies(scoped_refptr<Operator>& op);
    void sync_compute_dependencies(Operator* op);

    struct ThreadContext {
        Worker* worker;
        int thread_idx;
    };
    std::vector<ThreadContext> _compute_ctxs;
    std::vector<ThreadContext> _transfer_ctxs;
    // compute and transfer threads
    static void* dse_compute_thread_runtime(void* arg);
    static void* dse_transfer_thread_runtime(void* arg);
    // real work functions
    void do_compute_work(int thread_idx);
    void do_transfer_work(int thread_idx);

    std::vector<std::shared_ptr<OpQueue>> _buffered_compute_ops;
    std::vector<std::vector<scoped_refptr<Operator>>> _processing_compute_ops;
    
    std::vector<std::shared_ptr<OpQueue>> _buffered_transfer_ops;
    std::vector<std::vector<scoped_refptr<Operator>>> _processing_transfer_ops;

    WorkerJobExecutor _job_executor;
    uint32_t _worker_id;

    // device
    c10::Device _device;
    // stopped flag
    std::atomic<bool> _stopped;

    int _compute_thread_num;
    int _transfer_thread_num;
    std::vector<pthread_t> _compute_threads;
    std::vector<pthread_t> _transfer_threads;
};


}