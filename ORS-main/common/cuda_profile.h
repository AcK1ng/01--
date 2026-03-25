#include <torch/torch.h>
#include <torch/csrc/autograd/profiler.h>

namespace ors {

class NvtxProfilerGuard {
public:
    // 构造时开启 Profiler
    NvtxProfilerGuard(bool enabled = true) : enabled_(enabled) {
        if (enabled_) {
            // 配置：开启 NVTX 模式，记录 Shapes
            torch::autograd::profiler::ProfilerConfig config(
                torch::autograd::profiler::ProfilerState::NVTX, // 关键：指定 NVTX
                /*report_input_shapes=*/true,
                /*profile_memory=*/false
            );

            std::set<torch::autograd::profiler::ActivityType> activities = {
                torch::autograd::profiler::ActivityType::CPU,
                torch::autograd::profiler::ActivityType::CUDA
            };


            try {
                torch::autograd::profiler::enableProfiler(config, activities);
            } catch (const std::exception& e) {
                std::cerr << "[NvtxProfilerGuard] Failed to enable profiler: " 
                          << e.what() << std::endl;
                enabled_ = false; // 标记为未成功开启
            }
        }
    }

    // 析构时关闭
    ~NvtxProfilerGuard() {
        if (enabled_) {
            try {
                // 关闭并获取结果（NVTX 模式下结果直接发给 Nsight，这里返回的结果用于统计）
                torch::autograd::profiler::disableProfiler();
            } catch (const std::exception& e) {
                std::cerr << "Error disabling profiler: " << e.what() << std::endl;
            }
        }
    }

    // 禁止拷贝和赋值
    NvtxProfilerGuard(const NvtxProfilerGuard&) = delete;
    NvtxProfilerGuard& operator=(const NvtxProfilerGuard&) = delete;

private:
    bool enabled_;
};

}