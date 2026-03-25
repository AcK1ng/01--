#pragma once
#include <string>
#include <mutex>
#include <butil/status.h>
#include <butil/memory/ref_counted.h>
#include <butil/containers/flat_map.h>
#include "common/closure_helper.h"

namespace ors {

class WorkerJob {
public:
    WorkerJob() {}
    virtual ~WorkerJob() {}
    virtual butil::Status execute() = 0;
    virtual void on_finished(butil::Status& st) = 0;
    std::string _job_id;    
};

class WorkerJobExecutor {
public:
    WorkerJobExecutor();
    ~WorkerJobExecutor() {}

    void start();
    void stop();
    void join();
    void register_job(WorkerJob* context);
private:
    struct JobStatus : public butil::RefCountedThreadSafe<JobStatus> {
        JobStatus() {
            AddRef();
        }
        ~JobStatus() {
            context->on_finished(status);
        }
        WorkerJob* context = nullptr;
        bool flying = false;
        butil::Status status;
    };

    void add_new_job(JobStatus* job_status);
    void execute(JobStatus* job_status);

    butil::FlatMap<std::string, JobStatus*> _jobs;
    int64_t _flying_requests = 0;
    bool _stopped = false;
    std::mutex _mutex;

};


}