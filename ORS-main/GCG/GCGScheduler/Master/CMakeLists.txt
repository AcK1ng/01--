file(GLOB Master_SRCS
        *.cpp
        FullMaster/*.cpp
      )
file(GLOB Master_HEADERS
        *.h
        FullMaster/*.h
      )

add_library(Master STATIC ${Master_SRCS} ${Master_HEADERS})

