#pragma once
#include "master/operator.h"
#include "worker/engine.h"
#include "master/latency_estimate.h"
#include <mutex>

namespace ors {

extern LatencyEstimator g_latency_est;

class IScheduler {
public:
    IScheduler() {}
    virtual ~IScheduler() = default;
    virtual int start() = 0;
    virtual void stop() = 0;
    virtual bool schedule(std::vector<scoped_refptr<Operator>>& OpQueue) = 0;
};

class RoundRobinScheduler : public IScheduler {
public:
    RoundRobinScheduler(Engine* engine)
        : _engine(engine),
          _stopped(true) {}

    virtual ~RoundRobinScheduler() {
        if (!_stopped.load(std::memory_order_acquire)) {
            stop();
        }
    }

    int start() override;
    void stop() override;
    bool schedule(std::vector<scoped_refptr<Operator>>& OpQueue);
    void do_schedule_ops();

    static void* run(void* arg);
private:
    Engine* _engine;
    std::atomic<bool> _stopped;
    pthread_t _scheduler_thread;
    std::mutex _mutex;
};

class ProfileScheduler : public IScheduler {
public:
    ProfileScheduler(Engine* engine)
        : _engine(engine),
          _stopped(true) {}

    virtual ~ProfileScheduler() {
        if (!_stopped.load(std::memory_order_acquire)) {
            stop();
        }
    }

    int start() override;
    void stop() override;
    bool schedule(std::vector<scoped_refptr<Operator>>& OpQueue);
    void do_schedule_ops();
    void set_profile_worker_id(int worker_id) {
        std::lock_guard<std::mutex> lock(_mutex);
        _profile_worker_id = worker_id;
    }

    static void* run(void* arg);
private:
    Engine* _engine;
    std::atomic<bool> _stopped;
    pthread_t _scheduler_thread;
    int _profile_worker_id = 0;
    std::mutex _mutex;
};

class CostModel {
public:
    struct TransferParams {
        double threshold_bytes; // S_threshold
        double alpha;           // 小数据固定延迟
        double beta0;           // 大数据截距
        double beta1;           // 大数据斜率 (1/Bandwidth)
    };

    TransferParams pcie_params_ = {
        1080.08 * 1024.0, // Threshold: 64KB
        47.4,        // Alpha: 20us
        103.7,        // Beta0: 15us
        5.096458868178638e-05      // Beta1: 对应约 12GB/s
    };

    uint64_t predict_transfer(size_t size_bytes, int src_dev, int dst_dev) {
        if (src_dev == dst_dev) return 0;
        if (static_cast<double>(size_bytes) <= pcie_params_.threshold_bytes) {
            return pcie_params_.alpha;
        } else {
            return pcie_params_.beta0 + pcie_params_.beta1 * static_cast<double>(size_bytes);
        }
    }

    uint64_t predict_compute(std::string model_id, uint64_t op_id, int device_id) {
        OpKey key;
        key.model_id = model_id;
        key.op_id = op_id;
        OpProfileData profile = g_latency_est.GetProfiledData(key);
        return profile.latency_profile[device_id];
    }
};

struct WorkerState {
    int worker_id;
    int device_id;
    
    uint64_t transfer_finish_time = 0;
    uint64_t compute_finish_time = 0;

    uint64_t get_transfer_ready(uint64_t now) const {
        return std::max(now, transfer_finish_time);
    }

    uint64_t get_compute_ready(uint64_t now) const {
        return std::max(now, compute_finish_time);
    }
};

class HEFTScheduler : public IScheduler {
public:
    HEFTScheduler(Engine* engine, LatencyEstimator* latency_est)
        : _engine(engine),
          _latency_est(latency_est),
          _stopped(true) {
        for (auto& w : _engine->_all_workers) {
            _scoreboard[w->get_worker_id()] = {w->get_worker_id(), w->get_device_id(), 0, 0};
        }
    }

    virtual ~HEFTScheduler() {
        if (!_stopped.load(std::memory_order_acquire)) {
            stop();
        }
    }

    int start() override;
    void stop() override;
    bool schedule(std::vector<scoped_refptr<Operator>>& OpQueue);
    uint64_t estimate_transfer_cost(scoped_refptr<Operator>& op, int target_dev_id);
    uint64_t estimate_compute_cost(scoped_refptr<Operator>& op, int target_dev_id);


    std::pair<uint64_t, uint64_t> calculate_eft(scoped_refptr<Operator>& op, int dev_id, uint64_t now);
    void update_scoreboard(int dev_id, uint64_t eft, uint64_t t_move, uint64_t now);

    void do_schedule_ops();
    void get_optimal_worker(scoped_refptr<Operator>& op, uint64_t now);
    static void* run(void* arg);
private:
    Engine* _engine;
    LatencyEstimator* _latency_est;
    std::atomic<bool> _stopped;
    pthread_t _scheduler_thread;

    std::unordered_map<int, WorkerState> _scoreboard;
    // std::vector<WorkerState> _scoreboard;
    CostModel _cost_model;
   
    std::mutex _mutex;
};

}