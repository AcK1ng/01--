file(GLOB ClusterSys_SRCS
        *.cpp
      )
file(GLOB ClusterSys_HEADERS
        *.h
      )

add_library(ClusterSys STATIC ${ClusterSys_SRCS} ${ClusterSys_HEADERS})
