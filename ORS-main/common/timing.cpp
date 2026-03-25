#include "common/timing.h"
#include <butil/logging.h>
#include <iostream>  // 缺少这行会导致错误

namespace ors {

Timer::~Timer()
{
    this->LogResults();
}

void Timer::RecordBegin()
{
    CHECK(getrusage(RUSAGE_SELF, &usage_begin_) == 0) << "getrusage failed";
    begin_ = std::chrono::system_clock::now();
}

void Timer::RecordEnd()
{
    CHECK(getrusage(RUSAGE_SELF, &usage_end_) == 0) << "getrusage failed";
    end_ = std::chrono::system_clock::now();
    total_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - begin_).count();
    total_cpu_ns_ +=
        (GetCpuTimeUs(&usage_end_) - GetCpuTimeUs(&usage_begin_)) * 1000;
    total_cnt_++;
}

void Timer::LogResults()
{
    LOG(INFO) << "[TIMER] " << kName.c_str() 
              << "cnt " << total_cnt_ 
              << "avg" << this->GetAvgNs() / 1000.0 << "us";
}

int64_t Timer::GetAvgNs()
{
    if (total_cnt_ == 0) return 0;
    return total_ns_ / total_cnt_;
}

int64_t Timer::GetAvgNsCpu()
{
    if (total_cnt_ == 0) return 0;
    return total_cpu_ns_ / total_cnt_;
}

void Accumulator::Start()
{
    if (started_) return;
    begin_ = std::chrono::system_clock::now();
    started_ = true;
}

void Accumulator::Stop()
{
    if (!started_) return;
    accumulated_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - begin_).count();
    started_ = false;
}

void Accumulator::Reset()
{
    accumulated_ns_ = 0;
    started_ = false;
}

int64_t Accumulator::GetAccumulatedNs()
{
    Stop();
    return accumulated_ns_;
}

}