if (WITH_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

file(GLOB Cluster_SRCS
        *.cpp
        OOQueue/*.cpp
        RayCluster/*.cpp
      )
file(GLOB Cluster_HEADERS
        *.h
        OOQueue/*.h
        RayCluster/*.h
      )


add_library(Cluster STATIC ${Cluster_SRCS} ${Cluster_HEADERS})
