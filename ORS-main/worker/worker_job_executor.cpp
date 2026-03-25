#include "worker/worker_job_executor.h"
#include <bthread/errno.h>

namespace ors {

DEFINE_int32(ors_worker_job_executor_map_size, 64,
             "the initial size of WorkerJobExecutorMap");

WorkerJobExecutor::WorkerJobExecutor() {
    CHECK_EQ(0, _jobs.init(FLAGS_ors_worker_job_executor_map_size));
}

void WorkerJobExecutor::start() {
    std::unique_lock<std::mutex> lck(_mutex);
    _stopped = false;
}

void WorkerJobExecutor::stop() {
    std::vector<JobStatus*> deleted_jobs;
    std::unique_lock<std::mutex> lck(_mutex);
    _stopped = true;
    for (auto& job : _jobs) {
        LOG(ERROR) << "find undeleted jobs, job_id: " << job.second->context->_job_id;
        deleted_jobs.push_back(job.second);
        job.second->status = butil::Status(ESTOP, "Executor stopped");
    }
    _jobs.clear();
    lck.unlock();
    for (auto& job : deleted_jobs) {
        job->Release();
    }
}

void WorkerJobExecutor::join() {
    std::unique_lock<std::mutex> lck(_mutex);
    for (;;) {
        if (_flying_requests == 0) {
            break;
        }
        lck.unlock();
        usleep(1000);
        lck.lock();
    }
}

void WorkerJobExecutor::register_job(WorkerJob* context) {
    auto job_status = new JobStatus;
    job_status->context = context;
    return add_new_job(job_status);
}

void WorkerJobExecutor::add_new_job(JobStatus* job_status) {
    auto& job_id = job_status->context->_job_id;
    std::unique_lock<std::mutex> lck(_mutex);
    if (_stopped) {
        job_status->status = butil::Status(ESTOP, "Executor stopped");
        lck.unlock();
        job_status->Release();
        return;
    }
    if (_jobs.seek(job_id)) {
        job_status->status = butil::Status(EEXIST, "Job already exist");
        lck.unlock();
        LOG(ERROR) << "fail to add job " << job_id << ", already exist";
        job_status->Release();
        return;
    }
    _jobs[job_id] = job_status;
    job_status->flying = true;
    ++_flying_requests;
    lck.unlock();
    execute(job_status);
}

void WorkerJobExecutor::execute(JobStatus* job_status) {
    // auto execute_done = create_closure([job_status, this](butil::Status& st) {
    //     std::unique_lock<std::mutex> lck(_mutex);
    //     job_status->flying = false;
    //     --_flying_requests;
    //     CHECK(_flying_requests >= 0);
    //     job_status->status.swap(st);
    //     _jobs.erase(job_status->context->_job_id);
    //     lck.unlock();
    //     job_status->Release();
    // });
    // job_status->context->execute(execute_done);
    return;
}

}