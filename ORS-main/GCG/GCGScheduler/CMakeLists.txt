cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GCG)


include_directories("./include")


# Ray cmake template
# https://github.com/ray-project/ray/issues/31153

# https://pytorch.org/cppdocs/installing.html
execute_process(COMMAND python "-cimport torch;print(torch.utils.cmake_prefix_path)"
OUTPUT_VARIABLE Torch_ROOT
OUTPUT_STRIP_TRAILING_WHITESPACE
ERROR_STRIP_TRAILING_WHITESPACE
)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})


set(MY_LD_FLAGS "")


find_package(Python COMPONENTS Interpreter Development REQUIRED)

include_directories(${Python_INCLUDE_DIRS})
message(STATUS "Python_INCLUDE_DIRS = ${Python_INCLUDE_DIRS}")

list(APPEND MY_LD_FLAGS ${Python_LIBRARIES})


option(WITH_CUDA OFF)
if (WITH_CUDA)
    message(STATUS "Building with CUDA")
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DWITH_CUDA)
endif()

option(WITH_CANN OFF)
if (WITH_CANN)
    message(STATUS "Building with CANN")


    execute_process(COMMAND python "-cimport torch_npu;print(torch_npu.__path__[0]);"
    OUTPUT_VARIABLE Torch_npu_Root
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
    )

    include_directories("${Torch_npu_Root}/include")
    list(APPEND MY_LD_FLAGS "-L${Torch_npu_Root}/lib")
    list(APPEND MY_LD_FLAGS "-ltorch_npu")

    set(CANN_INSTALL_PATH $ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_CL_INCLUDE ${CANN_INSTALL_PATH}/include)
    set(ASCEND_CL_LIB ${CANN_INSTALL_PATH}/lib64)

    include_directories(${ASCEND_CL_INCLUDE})

    list(APPEND MY_LD_FLAGS "-L${ASCEND_CL_LIB}")
    list(APPEND MY_LD_FLAGS "-lascendcl")
    add_definitions(-DWITH_CANN)
endif()

# CXX11_ABI = 0 is required by Ray C++,
# so we re-build pytorch with _GLIBCXX_USE_CXX11_ABI=0
#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_link_options("-Wl,--no-as-needed")


# execute_process(
# COMMAND python "-cimport os;import ray;print(os.path.abspath(os.path.dirname(ray.__file__)))"
# OUTPUT_VARIABLE RAY_DIR
# OUTPUT_STRIP_TRAILING_WHITESPACE
# ERROR_STRIP_TRAILING_WHITESPACE
# )
# include_directories("${RAY_DIR}/cpp/include")

include_directories(${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory("Cluster")
add_subdirectory("Master")
add_subdirectory("ClusterSys")
add_subdirectory("Test")

add_library(GCGScheduler SHARED $<TARGET_OBJECTS:Cluster>
                         $<TARGET_OBJECTS:Master>
                         $<TARGET_OBJECTS:ClusterSys>)
target_link_libraries(GCGScheduler
        "${TORCH_LIBRARIES}"
        # "${RAY_DIR}/cpp/lib/libray_api.so"
        "${MY_LD_FLAGS}"
        hwloc
        )

