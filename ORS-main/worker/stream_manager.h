#pragma once
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <butil/memory/singleton.h>
#include <vector>

namespace ors {

class StreamManager {
DISALLOW_COPY_AND_ASSIGN(StreamManager);
    friend struct DefaultSingletonTraits<StreamManager>;
public:
    StreamManager() {}
    ~StreamManager() {}

    void init(int num_npus);

    static StreamManager* GetInstance() {
        return Singleton<StreamManager>::get();
    }

    c10_npu::NPUStream get_stream(int dev_id, int stream_id) {
        return _streams[dev_id][stream_id];
    }

private:
    bool _initialized = false;
    std::vector<std::vector<c10_npu::NPUStream>> _streams; // [dev_id][stream_id]
};

#define g_stream_manager StreamManager::GetInstance()

}
