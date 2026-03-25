cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(verify_scheme)

# 设置 LibTorch 路径（如果在环境变量里没设，可以在这里写死）
# set(CMAKE_PREFIX_PATH "/path/to/libtorch")

find_package(Torch REQUIRED PATHS "/usr/local/lib/libtorch")
find_package(CUDA REQUIRED)

# 添加可执行文件
add_executable(verify_scheme verify_scheme.cpp)

# 链接 Torch 和 CUDA 库
target_link_libraries(verify_scheme "${TORCH_LIBRARIES}")
target_link_libraries(verify_scheme ${CUDA_LIBRARIES})

# 为了使用 NVTX (nvToolsExt)，我们需要显式链接它
# 通常它位于 CUDA Toolkit 的 lib64 下
find_library(NVTX_LIB nvToolsExt PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
if(NVTX_LIB)
    target_link_libraries(verify_scheme ${NVTX_LIB})
else()
    message(WARNING "NVTX library not found, profiling markers might not work.")
endif()

# 设置 C++ 标准
set_property(TARGET verify_scheme PROPERTY CXX_STANDARD 17)
