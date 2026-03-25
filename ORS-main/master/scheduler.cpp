#include "master/scheduler.h"
#include <sys/sysinfo.h>
#include <nvToolsExt.h>


namespace ors {

int RoundRobinScheduler::start() {
    if (!_stopped.load(std::memory_order_acquire)) {
        LOG(WARNING) << "RoundRobinScheduler already started";
        return 0;
    }
    LOG(INFO) << "RoundRobinScheduler starting";
    _stopped.store(false, std::memory_order_release);
    if (pthread_create(&_scheduler_thread, nullptr, run, this)) {
        LOG(ERROR) << "create scheduler thread failed";
        return -1;
    }
    // bind core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int cpu_core_num = get_nprocs();
    srand(time(nullptr) ^ getpid());
    int bind_core_id = rand() % cpu_core_num;
    CPU_SET(bind_core_id, &cpuset);
    pthread_setaffinity_np(_scheduler_thread, sizeof(cpu_set_t), &cpuset);
    LOG(INFO) << "scheduler bind to core: " << bind_core_id;
    return 0;
}

void RoundRobinScheduler::stop() {
    if (_stopped.load(std::memory_order_acquire)) {
        LOG(WARNING) << "RoundRobinScheduler already stopped";
        return;
    }
    LOG(INFO) << "RoundRobinScheduler stopping";
    _stopped.store(true, std::memory_order_release);
    pthread_join(_scheduler_thread, NULL);
    LOG(INFO) << "RoundRobinScheduler stopped";
}

void* RoundRobinScheduler::run(void* arg) {
    RoundRobinScheduler* scheduler = static_cast<RoundRobinScheduler*>(arg);
    scheduler->do_schedule_ops();
    return nullptr;
}

void RoundRobinScheduler::do_schedule_ops() {
    // while (true) {
    //     if (_stopped.load(std::memory_order_acquire)) {
    //         LOG(INFO) << "RoundRobinScheduler stopped";
    //         break;
    //     }
    //     std::vector<scoped_refptr<Operator>> buffered_ops;
    //     buffered_ops.clear();
    //     g_rpool->batch_pop_operator(buffered_ops);
    //     if (!schedule(buffered_ops)) {
    //         break;
    //     }
    // }
    return;
}

bool RoundRobinScheduler::schedule(std::vector<scoped_refptr<Operator>>& OpQueue) {
    //只有一个GPU0进行调度
    static int worker_idx = 0;
    int num_workers = _engine->_all_workers.size();
    // std::cout << "num_workers: " << num_workers << std::endl;
    for (auto& op : OpQueue) {
        std::string tag = "schedule " + op->get_op_name() + "_" + std::to_string(op->get_graph_id()) + "_" + std::to_string(op->id()) + " to worker " + std::to_string(worker_idx);
        nvtxRangePush(tag.c_str());
        auto worker = _engine->_all_workers[worker_idx];
        CHECK(worker != nullptr);
        worker->enqueue_transfer(op);
        // 打印日志方便调试调度情况
        // LOG(ERROR) << "Scheduled op " << op->get_op_name() << " to Worker " << worker_idx;
        // 轮询下一个 Worker
        worker_idx = (worker_idx + 1) % num_workers;
        nvtxRangePop();
    }
    return true;
}
//----------------------------------------------------------------------------------------------------//
int ProfileScheduler::start() {
    if (!_stopped.load(std::memory_order_acquire)) {
        LOG(WARNING) << "ProfileScheduler already started";
        return 0;
    }
    LOG(INFO) << "ProfileScheduler starting";
    _stopped.store(false, std::memory_order_release);
    if (pthread_create(&_scheduler_thread, nullptr, run, this)) {
        LOG(ERROR) << "create scheduler thread failed";
        return -1;
    }
    // bind core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int cpu_core_num = get_nprocs();
    srand(time(nullptr) ^ getpid());
    int bind_core_id = rand() % cpu_core_num;
    CPU_SET(bind_core_id, &cpuset);
    pthread_setaffinity_np(_scheduler_thread, sizeof(cpu_set_t), &cpuset);
    LOG(INFO) << "scheduler bind to core: " << bind_core_id;
    return 0;
}

void ProfileScheduler::stop() {
    if (_stopped.load(std::memory_order_acquire)) {
        LOG(WARNING) << "ProfileScheduler already stopped";
        return;
    }
    LOG(INFO) << "ProfileScheduler stopping";
    _stopped.store(true, std::memory_order_release);
    pthread_join(_scheduler_thread, NULL);
    LOG(INFO) << "ProfileScheduler stopped";
}

void* ProfileScheduler::run(void* arg) {
    ProfileScheduler* scheduler = static_cast<ProfileScheduler*>(arg);
    scheduler->do_schedule_ops();
    return nullptr;
}

void ProfileScheduler::do_schedule_ops() {
    // while (true) {
    //     if (_stopped.load(std::memory_order_acquire)) {
    //         LOG(INFO) << "ProfileScheduler stopped";
    //         break;
    //     }
    //     std::vector<scoped_refptr<Operator>> buffered_ops;
    //     buffered_ops.clear();
    //     g_rpool->batch_pop_operator(buffered_ops);
    //     if (!schedule(buffered_ops)) {
    //         break;
    //     }
    // }
    return;
}

bool ProfileScheduler::schedule(std::vector<scoped_refptr<Operator>>& OpQueue) {
    for (auto& op : OpQueue) {
        auto worker = _engine->_all_workers[_profile_worker_id];
        CHECK(worker != nullptr);
        worker->enqueue_transfer(op);
    }
    return true;
}

//----------------------------------------------------------------------------------------------------//
int HEFTScheduler::start() {
    if (!_stopped.load(std::memory_order_acquire)) {
        LOG(WARNING) << "HEFTScheduler already started";
        return 0;
    }
    LOG(INFO) << "HEFTScheduler starting";
    _stopped.store(false, std::memory_order_release);
    if (pthread_create(&_scheduler_thread, nullptr, run, this)) {
        LOG(ERROR) << "create scheduler thread failed";
        return -1;
    }
    // bind core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int cpu_core_num = get_nprocs();
    srand(time(nullptr) ^ getpid());
    int bind_core_id = rand() % cpu_core_num;
    CPU_SET(bind_core_id, &cpuset);
    pthread_setaffinity_np(_scheduler_thread, sizeof(cpu_set_t), &cpuset);
    LOG(INFO) << "scheduler bind to core: " << bind_core_id;
    return 0;
}

void HEFTScheduler::stop() {
    if (_stopped.load(std::memory_order_acquire)) {
        LOG(WARNING) << "RoundRobinScheduler already stopped";
        return;
    }
    LOG(INFO) << "HEFTScheduler stopping";
    _stopped.store(true, std::memory_order_release);
    pthread_join(_scheduler_thread, NULL);
    LOG(INFO) << "HEFTScheduler stopped";
}

void* HEFTScheduler::run(void* arg) {
    HEFTScheduler* scheduler = static_cast<HEFTScheduler*>(arg);
    scheduler->do_schedule_ops();
    return nullptr;
}

uint64_t HEFTScheduler::estimate_transfer_cost(scoped_refptr<Operator>& op, int target_dev_id) {
    // uint64_t total_cost = 0;
    // for (const auto& kv : op->_input_meta) {
    //     const TensorMeta& meta = kv.second;
    //     for (const auto& dist : meta.distribution) {
    //         int src_dev_id = dist.first;
    //         size_t size_bytes = dist.second;
    //         if (src_dev_id != target_dev_id) {
    //             total_cost += _cost_model.predict_transfer(
    //                 size_bytes, 
    //                 src_dev_id, 
    //                 target_dev_id
    //             );
    //         }
    //     }
    // }
    // return total_cost;
    return 0;
}

uint64_t HEFTScheduler::estimate_compute_cost(scoped_refptr<Operator>& op, int target_dev_id) {
    uint64_t cost = _cost_model.predict_compute(op->_model_id, op->_op_id, target_dev_id);
    return cost;
}

void HEFTScheduler::do_schedule_ops() {
    return;
    // while (true) {
    //     if (_stopped.load(std::memory_order_acquire)) {
    //         LOG(INFO) << "HEFTScheduler stopped";
    //         break;
    //     }
    //     std::vector<scoped_refptr<Operator>> buffered_ops;
    //     buffered_ops.clear();
    //     g_rpool->batch_pop_operator(buffered_ops);
    //     if (!schedule(buffered_ops)) {
    //         break;
    //     }
    // }
}

bool HEFTScheduler::schedule(std::vector<scoped_refptr<Operator>>& OpQueue) {
    for (auto& op : OpQueue) {
        uint64_t now = butil::cpuwide_time_us();
        get_optimal_worker(op, now);
    }
    return true;
}

std::pair<uint64_t, uint64_t> HEFTScheduler::calculate_eft(scoped_refptr<Operator>& op, int worker_id, uint64_t now) {
    auto& state = _scoreboard[worker_id];
    uint64_t t_move;
    if (op->is_cpu_op()) {
        t_move = 0;
    } else {
        t_move = estimate_transfer_cost(op, state.device_id);
    }
    uint64_t t_exec = estimate_compute_cost(op, state.device_id);

    uint64_t data_ready_time = state.get_transfer_ready(now) + t_move;
    uint64_t compute_ready_time = state.get_compute_ready(now);

    uint64_t actual_start_time = std::max(data_ready_time, compute_ready_time);
    uint64_t finish_time = actual_start_time + t_exec;

    return {finish_time, t_move};
}

void HEFTScheduler::update_scoreboard(int dev_id, uint64_t eft, uint64_t t_move, uint64_t now) {
    auto& state = _scoreboard[dev_id];

    if (t_move > 0) {
        uint64_t transfer_start = std::max(now, state.transfer_finish_time);
        // 新的水位线 = 开始时间 + 传输耗时
        state.transfer_finish_time = transfer_start + t_move;
    }
    state.compute_finish_time = eft;
}

void HEFTScheduler::get_optimal_worker(scoped_refptr<Operator>& op, uint64_t now) {
    return;
    // int best_dev_id = -1;
    // int best_worker_id = -1;
    // uint64_t min_eft = UINT64_MAX;
    // uint64_t best_move = 0;
    // // 如果是CPU算子直接就分发了
    // if (op->is_cpu_op() && _engine->get_cpu_workers().size() > 0) {
    //     for (auto& w : _engine->get_cpu_workers()) {
    //         auto result = calculate_eft(op, w->get_worker_id(), now);
    //         uint64_t eft = result.first;
    //         uint64_t t_move = result.second;

    //         auto& state = _scoreboard[w->get_worker_id()];
    //         if (state.compute_finish_time <= now) {
    //             min_eft = eft;
    //             best_dev_id = w->get_device_id();
    //             best_worker_id = w->get_worker_id();
    //             best_move = 0;
    //             break;
    //         } else if (eft < min_eft) {
    //             min_eft = eft;
    //             best_dev_id = w->get_device_id();
    //             best_worker_id = w->get_worker_id();
    //             best_move = t_move;
    //         }
    //     }
    //     CHECK(best_worker_id != -1);
    //     update_scoreboard(best_worker_id, min_eft, best_move, now);
    //     _engine->_all_workers[best_worker_id]->enqueue_transfer(op);
    // } else {
    //     // 如果是必须放在GPU上执行的，则需要进行调度
    //     for (auto& w : _engine->get_gpu_workers()) {
    //         auto result = calculate_eft(op, w->get_worker_id(), now);
    //         uint64_t eft = result.first;
    //         uint64_t t_move = result.second;

    //         if (eft < min_eft) {
    //             min_eft = eft;
    //             best_dev_id = w->get_device_id();
    //             best_worker_id = w->get_worker_id();
    //             best_move = t_move;
    //         }
    //     }
    //     CHECK(best_worker_id != -1);
    //     update_scoreboard(best_worker_id, min_eft, best_move, now);
    //     _engine->_all_workers[best_worker_id]->enqueue_transfer(op);
    // }
}

}