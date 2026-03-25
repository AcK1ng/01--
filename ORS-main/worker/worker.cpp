#include "worker/worker.h"
#include <butil/logging.h>
#include <sys/sysinfo.h>
#include "master/rpool.h"
#include "common/utils.h"
#include <gflags/gflags.h>
#include <random>
#include <torch/torch.h>
#include "worker/stream_manager.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace ors {

DEFINE_bool(worker_dse_bind_core, true, "worker dse bind core");

thread_local c10::Device torch_device("cpu");
void set_torch_device__thread(c10::Device device) {
    torch_device = device;
}

torch::RegisterOperators native_device_getter(
        "prim::GCG_get_native_device",
        []() {return torch_device;});

thread_local Operator* dse_target = nullptr;
thread_local int dse_thread_idx = -1;

Worker::Worker(uint32_t worker_id,
               c10::DeviceType dev_type,
               c10::DeviceIndex dev_index) : 
                _worker_id(worker_id),
                _device(c10::Device(dev_type, dev_index)),
                _stopped(true) {
    _compute_thread_num = 8;
}

int Worker::start() {
    if (!_stopped.load(std::memory_order_acquire)) {
        LOG(WARNING) << "Worker already started";
        return 0;
    }
    LOG(INFO) << "worker starting";
    _job_executor.start();
    _stopped.store(false, std::memory_order_release);

    _compute_ctxs.resize(_compute_thread_num);
    _compute_threads.reserve(_compute_thread_num);
    for (int i = 0; i < _compute_thread_num; ++i) {
        pthread_t tid;
        _compute_ctxs[i] = {this, i};
        if (pthread_create(&tid, nullptr, dse_compute_thread_runtime, &_compute_ctxs[i])) {
            LOG(ERROR) << "create dse compute thread failed";
            return -1;
        }
        _compute_threads.push_back(tid);

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i * 2 + 1, &cpuset);
        pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
    }

    LOG(INFO) << "worker started";
    return 0;
}

void Worker::stop() {
    if (_stopped.load(std::memory_order_acquire)) {
        LOG(WARNING) << "worker already stopped";
        return;
    }

    LOG(INFO) << "dse thread stopping";
    _stopped.store(true, std::memory_order_release);
    for (size_t i = 0; i < _compute_threads.size(); ++i) {
        pthread_join(_compute_threads[i], NULL);
    }
    for (size_t i = 0; i < _transfer_threads.size(); ++i) {
        pthread_join(_transfer_threads[i], NULL);
    }
    LOG(INFO) << "des thread stopped";

    LOG(INFO) << "worker job executor stopping";
    _job_executor.stop();
    _job_executor.join();
    LOG(INFO) << "worker job executor stopped";
}

void Worker::enqueue_compute(scoped_refptr<Operator>& op) {
    _buffered_compute_ops[op->id() % _compute_thread_num]->push(op);
}

void Worker::enqueue_transfer(scoped_refptr<Operator>& op) {
    _buffered_transfer_ops[op->id() % _transfer_thread_num]->push(op);
}

void Worker::do_compute_work(int thread_idx) {
    std::vector<Operator*> fetch_ops;
    auto tmp_g_rpool = Rpool::GetInstance();
    butil::Status st;
    auto current_stream = c10_npu::getCurrentNPUStream(_device.index());
    while (true) {
        if (_stopped.load(std::memory_order_acquire)) {
            break;
        }

        if (dse_target == nullptr) {
            int num_fetch = tmp_g_rpool->rpool_fetch_operator(fetch_ops, 1, dse_thread_idx);
            if (num_fetch > 0) {
                dse_target = fetch_ops[0];
                fetch_ops.clear();
            }
        }
        if (dse_target == nullptr) {
            continue;
        }

        auto op = dse_target;
        dse_target = nullptr;
        op->set_device(_device);

        op->_assigned_stream = current_stream;
        // sync stream dependencies
        sync_compute_dependencies(op);
        // execute operator
        st = op->execute();
        op->on_finished(st);
    }
}

void* Worker::dse_compute_thread_runtime(void* arg) {
    LOG(INFO) << "start dse compute thread runtime";
    ThreadContext* ctx = static_cast<ThreadContext*>(arg);
    Worker* worker = ctx->worker;
    dse_thread_idx = ctx->thread_idx;
    if (worker->_device.type() == c10::kPrivateUse1) {
        // NPU: bind device and stream
        c10_npu::NPUGuard device_guard(worker->_device);
        c10_npu::NPUStreamGuard stream_guard(g_stream_manager->get_stream(
            worker->_device.index(), ctx->thread_idx));
        set_torch_device__thread(worker->_device);
        c10::InferenceMode guard;
        worker->do_compute_work(ctx->thread_idx);
    } else {
        CHECK(false);
        set_torch_device__thread(worker->_device);
        // TODO: CPU compute worker
        worker->do_compute_work(ctx->thread_idx);
    }
    return nullptr;
}

void Worker::sync_transfer_dependency_recursive(
    Operator* parent,
    Operator* current_op,
    std::unordered_set<Operator*>& visited
) {
    return;
}

void Worker::sync_transfer_dependencies(scoped_refptr<Operator>& op) {
    std::unordered_set<Operator*> visited;
    for (auto& parent : op->get_parent_ops()) {
        sync_transfer_dependency_recursive(parent, op.get(), visited);
    }
}

void force_global_synchronize_for_debug() {
    for (int dev_idx = 0; dev_idx < 2; ++dev_idx) {
        c10_npu::NPUGuard g(at::Device(at::DeviceType::PrivateUse1, dev_idx));
        c10_npu::npuSynchronizeDevice();
    }
}

void Worker::sync_compute_dependencies(Operator* op) {
    for (auto& parent : op->get_parent_ops()) {
        if (parent->_assigned_stream.has_value() &&
            op->_assigned_stream.has_value() &&
            parent->_assigned_stream != op->_assigned_stream) {
            if (!parent->synchronizable()) {
                parent->enable_synchronization();
            }
            parent->synchronize(op->_assigned_stream.value());
        }
    }
}

void Worker::do_transfer_work(int thread_idx) {
    return;
}

void* Worker::dse_transfer_thread_runtime(void* arg) {
    return nullptr;
}

}
