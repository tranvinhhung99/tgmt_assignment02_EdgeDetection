cmake_minimum_required(VERSION 2.8)
project( EdgeDetection )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( Source )

add_subdirectory( Source )

if( CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory( Test )
    add_subdirectory( lib/googletest )
endif()
