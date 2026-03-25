#include <master/rpool.h>

namespace ors {

// void Rpool::push_operator(scoped_refptr<Operator> op) {
//     std::unique_lock<std::mutex> lck(_mutex);
//     _rpool_queue.push_back(op);
// }

// void Rpool::batch_push_operator(std::vector<scoped_refptr<Operator>>& ops) {
//     if (ops.empty()) return;
//     std::unique_lock<std::mutex> lck(_mutex);
//     _rpool_queue.insert(_rpool_queue.end(), ops.begin(), ops.end());
// }

// void Rpool::batch_pop_operator(std::vector<scoped_refptr<Operator>>& ops) {
//     std::unique_lock<std::mutex> lck(_mutex);
//     _rpool_queue.swap(ops);
// }

unsigned int Rpool::rpool_fetch_operator(std::vector<Operator*>& ops, 
                                unsigned int max_op_to_fetch, unsigned int dse_idx) {
    if (max_op_to_fetch == 0) return 0;
    rpool_queue* rpool_queue = nullptr;
    rpool_queue = get_queue_for_fetching(dse_idx);
    if (rpool_queue == nullptr) return 0;
    unsigned int num_ops = rpool_queue->pop_ops_from_queue(ops, max_op_to_fetch);
    rpool_queue->_occupied_mutex.unlock();
    return num_ops;
}

void Rpool::rpool_push_operator(std::vector<Operator*>& ops, 
                                unsigned int num_ops, unsigned int dse_idx) {
    if (num_ops == 0) return;
    rpool_queue* rpool_queue = nullptr;
    rpool_queue = get_queue_for_storing(dse_idx);
    if (rpool_queue == nullptr) return;
    rpool_queue->push_ops_to_queue(ops, num_ops);
    rpool_queue->_occupied_mutex.unlock();
}


} // namespace ors