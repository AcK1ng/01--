#include "worker/engine.h"
#include <Python.h>
#include <acl/acl.h>
#include "worker/stream_manager.h"
#include "common/event_pool.h"
#include <torch/csrc/autograd/profiler.h>


namespace ors {

// ACL error checking helper (mirrors GCG's npuErrchk pattern)
inline void npuAssert(aclError code, const char* file, int line, bool abort = true) {
    if (code != ACL_ERROR_NONE) {
        LOG(ERROR) << "ACL error " << code << " at " << file << ":" << line;
        if (abort) exit(code);
    }
}
#define npuErrchk(ans) { npuAssert((ans), __FILE__, __LINE__); }

int Engine::start() {
    LOG(INFO) << "engine starting";

    // Must import torch and torch_npu via Python before ACL init,
    // otherwise NPU devices won't be properly initialized (same as GCG).
    Py_Initialize();
    if (!Py_IsInitialized()) {
        LOG(ERROR) << "Python initialization failed";
        return -1;
    }
    int ret = PyRun_SimpleString(
        "import torch\n"
        "import torch_npu\n"
    );
    if (ret != 0) {
        LOG(ERROR) << "Failed to import torch / torch_npu via Python";
        return ret;
    }

    npuErrchk(aclInit(nullptr));

    uint32_t device_count = 0;
    aclError err = aclrtGetDeviceCount(&device_count);
    if (err != ACL_ERROR_NONE) {
        LOG(ERROR) << "ACL error: failed to get device count, code=" << err;
        return -1;
    }
    LOG(INFO) << "Detected " << device_count << " CANN devices";

    // init streams and event pool for all devices
    g_stream_manager->init(device_count);
    g_event_pool->init(device_count);
    g_rpool->add_queue(10);
    // npu worker
    int gpu_count = 1;
    for (int i = 0; i < gpu_count; ++i) {
        auto worker = std::make_shared<Worker>(i, c10::kPrivateUse1, i);
        LOG(INFO) << "npu worker " << i << " is created";
        worker->start();
        _gpu_workers.push_back(worker);
        _all_workers.push_back(worker);
    }

    LOG(INFO) << "engine started";
    return 0;
}

void Engine::stop() {
    LOG(INFO) << "engine stopping";
    for (size_t i = 0; i < _all_workers.size(); ++i) {
        _all_workers[i]->stop();
    }
    LOG(INFO) << "engine stopped";
    return;
}

void Engine::reset() {
    _all_workers.clear();
    _cpu_workers.clear();
    _gpu_workers.clear();
}

}
