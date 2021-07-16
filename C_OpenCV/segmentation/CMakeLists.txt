cmake_minimum_required (VERSION 2.8)
project (segmentation)
find_package( OpenCV REQUIRED )

set( NAME_SRC
	segmentation.cpp
	src/algorithm.cpp
	src/image.cpp
	src/cell.cpp
	src/roi.cpp
	src/watershed.cpp
	src/noise.cpp
	src/trackbar.cpp
	src/threshold.cpp
	src/feature.cpp
	src/colour.cpp
	src/morph.cpp
	src/kmeans.cpp
	src/fibre.cpp
)

set( NAME_HEADERS       
     inc/segmentation.hpp
     inc/cell.hpp
)

include_directories( ${OpenCV_INCLUDE_DIRS})
add_executable(segmentation ${NAME_SRC} ${NAME_HEADERS})

target_link_libraries( segmentation ${OpenCV_LIBS} )