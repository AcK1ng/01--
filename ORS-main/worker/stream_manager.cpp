#include "worker/stream_manager.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace ors {

void StreamManager::init(int num_npus) {
    if (_initialized) return;

    for (int i = 0; i < num_npus; ++i) {
        c10_npu::NPUGuard g(i);
        _streams.push_back({});
        for (int j = 0; j < 32; ++j) {
            _streams[i].push_back(c10_npu::getStreamFromPool(false, i));
        }
    }
    _initialized = true;
}

}
