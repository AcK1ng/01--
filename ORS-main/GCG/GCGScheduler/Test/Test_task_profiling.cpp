#include "Master/FullMaster/PerfPredictor.h"

int main() {
    auto task_performance_predictor = GetSimpleTaskPerformancePredictor(3, 0.2);
    task_performance_predictor->new_task(0, 5);
    for (size_t i = 0; i < 10; i++)
       task_performance_predictor->train(0, 3, "A100", 4000);
    auto res1 = task_performance_predictor->predict(0, 3, "A100");

    for (size_t i = 0; i < 10; i++)
       task_performance_predictor->train(0, 3, "A800", 2000);
    auto res2 = task_performance_predictor->predict(0, 3, "A800");

    for (size_t i = 0; i < 10; i++)
       task_performance_predictor->train(0, 3, "A100", 1000);
    auto res3 = task_performance_predictor->predict(0, 3, "A100");

    task_performance_predictor->drop_task(0);
    std::cout << res1 << std::endl;
    std::cout << res2 << std::endl;
    std::cout << res3 << std::endl;
    return 0;
}
