#pragma once
#include <vector>
#include <memory>
#include "worker/worker.h"
#include "common/type.h"

namespace ors {

class Engine {
public:
    friend class HEFTScheduler;
    friend class RoundRobinScheduler;
    friend class ProfileScheduler;
    Engine() {}
    ~Engine() {}

    int start();
    void stop();
    void reset();

    size_t GetNumWorkers() const {return _all_workers.size();}
    std::vector<std::shared_ptr<Worker>> get_cpu_workers() {
        return _cpu_workers;
    }
    std::vector<std::shared_ptr<Worker>> get_gpu_workers() {
        return _gpu_workers;
    }

    Worker* GetWorker(uint32_t id) {
        if (id >= 0 && id < _all_workers.size()) {
            return _all_workers[id].get();
        } else {
            return nullptr;
        }
    }

private:
    std::vector<std::shared_ptr<Worker>> _all_workers;
    std::vector<std::shared_ptr<Worker>> _cpu_workers;
    std::vector<std::shared_ptr<Worker>> _gpu_workers;
};

}