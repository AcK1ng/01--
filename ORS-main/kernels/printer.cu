// #include <cuda_runtime.h>
#include <stdio.h>
#include "kernels/printer.h"

template <typename T>
__global__ void print_kernel(const T* data, int num_elements, const char* prefix) {
    // 只让第一个线程打印，避免刷屏
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // printf("[GPU Print] %s: ", prefix);
        // 打印前 10 个元素
        // int count = (num_elements < 10) ? num_elements : 10;
        // for (int i = 0; i < count; ++i) {
        //     // 注意：CUDA printf 对 float 支持较好，其他类型可能需要强转
        //     printf("%f, ", (float)data[i]);
        // }
        printf("[GPU Print]");
        printf("\n");
    }
}

// 暴露给 C++ 的启动函数
void launch_gpu_print(const float* data, int num_elements, cudaStream_t stream, const char* msg) {
    print_kernel<<<1, 1, 0, stream>>>(data, num_elements, msg);
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("!!! GPU PRINT KERNEL FAILED !!!\n");
        printf("Error: %s\n", cudaGetErrorString(err));
        printf("Message: %s\n", msg);
        // 可以选择在这里 abort 或者抛出异常
        abort(); 
    }
}