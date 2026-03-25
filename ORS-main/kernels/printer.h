#pragma once
#include <cuda_runtime.h>
void launch_gpu_print(const float* data, int num_elements, cudaStream_t stream, const char* msg);