#include "event_pool.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace ors {

void EventPool::init(int num_devices) {
    pools_.resize(num_devices);
}

aclrtEvent EventPool::pop(c10::DeviceIndex device) {
    auto& pool = pools_[device];
    // Try to acquire an event from the per-device pool
    {
        std::lock_guard<std::mutex> g(pool.mutex_);
        if (!pool.event_pool_.empty()) {
            auto event = pool.event_pool_.back();
            pool.event_pool_.pop_back();
            return event;
        }
    }
    // If the pool is empty, create a new event on the target device.
    // ACL_EVENT_CAPTURE_STREAM_PROGRESS is required for aclrtStreamWaitEvent
    // (cross-stream synchronization), matching the original CUDA usage.
    aclrtEvent event;
    c10_npu::NPUGuard guard(device);
    aclrtCreateEventExWithFlag(&event, ACL_EVENT_CAPTURE_STREAM_PROGRESS);
    return event;
}

void EventPool::push(c10::DeviceIndex device, aclrtEvent event) {
    auto& pool = pools_[device];
    std::lock_guard<std::mutex> g(pool.mutex_);
    pool.event_pool_.push_back(event);
}

}
