#pragma once
#include <condition_variable>
#include <mutex>
#include <butil/memory/ref_counted.h>
#include <butil/memory/singleton.h>
#include "master/operator.h"

namespace ors {

struct rpool_queue {
    std::deque<Operator*> _ops;
    std::mutex _occupied_mutex;
    unsigned int num_stored;

    unsigned int pop_ops_from_queue(std::vector<Operator*>& ops, 
        unsigned int max_op_to_fetch) {

        unsigned int num_ops = 0;
        while (!_ops.empty() && num_ops < max_op_to_fetch) {
            ops.push_back(_ops.front());
            _ops.pop_front();
            num_ops++;
        }
        num_stored -= num_ops;
        return num_ops;
    }
    
    void push_ops_to_queue(std::vector<Operator*>& ops, 
        unsigned int num_ops) {
        for (unsigned int i = 0; i < num_ops; ++i) {
            _ops.push_back(ops[i]);
        }
        num_stored += num_ops;
    }
    
};

class Rpool {
DISALLOW_COPY_AND_ASSIGN(Rpool);
    friend struct DefaultSingletonTraits<Rpool>;
public:
    Rpool() {}
    ~Rpool() {}

    static Rpool* GetInstance() {
        return Singleton<Rpool>::get();
    }

    rpool_queue* get_queue_for_fetching(unsigned int dse_idx) {
        unsigned int queue_idx = dse_idx % _queue_arr.size();
        for (size_t i = 0; i < _queue_arr.size(); ++i) {
            if (_queue_arr[queue_idx]->num_stored > 0) {
                if (_queue_arr[queue_idx]->_occupied_mutex.try_lock()) {
                    return _queue_arr[queue_idx];
                }
            }
            queue_idx = (queue_idx + 1) % _queue_arr.size();
        }
        return nullptr;
    }
    
    rpool_queue* get_queue_for_storing(unsigned int dse_idx) {
        unsigned int queue_idx = dse_idx % _queue_arr.size();
        rpool_queue* target_queue = nullptr;
        while (target_queue == nullptr) {
            if (_queue_arr[queue_idx]->_occupied_mutex.try_lock()) {
                target_queue = _queue_arr[queue_idx];
            } else {
                queue_idx = (queue_idx + 1) % _queue_arr.size();
            }

        }
        return target_queue;
    }

    unsigned int rpool_fetch_operator(std::vector<Operator*>& ops, 
                                      unsigned int max_op_to_fetch, unsigned int dse_idx);
    void rpool_push_operator(std::vector<Operator*>& ops, 
        unsigned int num_ops, unsigned int dse_idx);

    void add_queue(int num_queues) {
        std::unique_lock<std::mutex> lck(_mutex);
        for (int i = 0; i < num_queues; ++i) {
            auto queue = new rpool_queue();
            // auto queue = std::make_shared<rpool_queue>();
            queue->num_stored = 0;
            _queue_arr.push_back(queue);
        }
    }
    // void push_operator(scoped_refptr<Operator> op);
    // void batch_push_operator(std::vector<scoped_refptr<Operator>>& ops);
    // void batch_pop_operator(std::vector<scoped_refptr<Operator>>& ops);

private:
    std::mutex _mutex;
    std::vector<rpool_queue*> _queue_arr;
};

#define g_rpool Rpool::GetInstance()

} // namespace ors