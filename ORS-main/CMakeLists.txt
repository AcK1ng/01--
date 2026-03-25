cmake_minimum_required(VERSION 3.12)
project(ORS VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Wall")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "$ENV{CXXFLAGS} -O2 -g -Wall")
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "Build types" FORCE)

option(ENABLE_TIME_DEBUG "Enable time debug" OFF)
if(ENABLE_TIME_DEBUG)
    message(STATUS ">>> ENABLE_TIME_DEBUG is ON")
    add_compile_definitions(ENABLE_TIME_DEBUG)
endif()

option(ENABLE_PROFILE_DATA "Enable profile data" OFF)
if(ENABLE_PROFILE_DATA)
    message(STATUS ">>> ENABLE_PROFILE_DATA is ON")
    add_compile_definitions(ENABLE_PROFILE_DATA)
endif()

# Ascend CANN support
option(WITH_CANN "Build with Ascend CANN" OFF)
if(WITH_CANN)
    message(STATUS "Building with Ascend CANN")

    # Get torch_npu installation path from Python
    execute_process(
        COMMAND python3 "-c" "import torch_npu; print(torch_npu.__path__[0])"
        OUTPUT_VARIABLE Torch_npu_Root
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "torch_npu root: ${Torch_npu_Root}")
    include_directories("${Torch_npu_Root}/include")
    link_directories("${Torch_npu_Root}/lib")

    # Ascend toolkit (set ASCEND_TOOLKIT_HOME env var before building)
    set(CANN_INSTALL_PATH $ENV{ASCEND_TOOLKIT_HOME})
    if(NOT CANN_INSTALL_PATH)
        message(FATAL_ERROR "ASCEND_TOOLKIT_HOME environment variable is not set")
    endif()
    set(ASCEND_CL_INCLUDE "${CANN_INSTALL_PATH}/include")
    set(ASCEND_CL_LIB "${CANN_INSTALL_PATH}/lib64")
    include_directories(${ASCEND_CL_INCLUDE})
    link_directories(${ASCEND_CL_LIB})

    # Python is required to call `import torch_npu` at runtime for NPU init
    find_package(Python3 REQUIRED COMPONENTS Development)
    include_directories(${Python3_INCLUDE_DIRS})

    add_definitions(-DWITH_CANN)
    message(STATUS "CANN include: ${ASCEND_CL_INCLUDE}")
    message(STATUS "CANN lib: ${ASCEND_CL_LIB}")
    message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python3 libs: ${Python3_LIBRARIES}")
endif()

# package
find_package(Torch REQUIRED PATHS "/usr/local/lib/libtorch")

# 定义源文件路径
file(GLOB WORKER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/worker/*.cpp")
file(GLOB COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp")
file(GLOB MASTER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/master/*.cpp")
file(GLOB MAIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main/*.cpp")
file(GLOB BENCHMARK_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/*/src/*.cpp")

# 定义头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories("/usr/local/lib/libtorch/include")
include_directories("/usr/include/jsoncpp")

# 生成ors的静态库
add_library(ors STATIC ${WORKER_SOURCES} ${COMMON_SOURCES} ${MASTER_SOURCES})

# 链接库
target_link_libraries(ors
    brpc
    crypto
    dl
    ssl
    protobuf
    glog
    gflags
    leveldb
    ${TORCH_LIBRARIES}
    jsoncpp
)

if(WITH_CANN)
    target_link_libraries(ors torch_npu ascendcl ${Python3_LIBRARIES})
endif()

foreach(MAIN_FILE ${MAIN_SOURCES})
    get_filename_component(FILE_NAME ${MAIN_FILE} NAME)
    string(REPLACE "_main.cpp" "" PREFIX ${FILE_NAME})
    message(STATUS "Processing main_file=${MAIN_FILE}, PREFIX=${PREFIX}")
    add_executable(${PREFIX} ${MAIN_FILE})
    target_link_libraries(${PREFIX} ors)
endforeach()

foreach(BENCHMARK_FILE ${BENCHMARK_SOURCES})
    get_filename_component(FILE_NAME ${BENCHMARK_FILE} NAME)
    string(REPLACE "_main.cpp" "" PREFIX ${FILE_NAME})
    message(STATUS "Processing benchmark_file=${BENCHMARK_FILE}, PREFIX=${PREFIX}")
    add_executable(${PREFIX} ${BENCHMARK_FILE})
    target_link_libraries(${PREFIX} ors)
endforeach()

