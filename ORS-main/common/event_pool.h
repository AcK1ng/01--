#pragma once
#include <vector>
#include <mutex>
#include <acl/acl.h>
#include <butil/memory/singleton.h>

namespace ors {

class EventPool {
DISALLOW_COPY_AND_ASSIGN(EventPool);
friend struct DefaultSingletonTraits<EventPool>;
public:
    static EventPool* GetInstance() {
        return Singleton<EventPool>::get();
    }
    EventPool() {}
    ~EventPool() {}

    // Must be called after aclInit() and aclrtGetDeviceCount()
    void init(int num_devices);

    aclrtEvent pop(c10::DeviceIndex device);
    void push(c10::DeviceIndex device, aclrtEvent event);
private:
    struct PerDevicePool {
        alignas(64) std::mutex mutex_;
        std::vector<aclrtEvent> event_pool_;
    };
    std::vector<PerDevicePool> pools_;
};

#define g_event_pool EventPool::GetInstance()

}
