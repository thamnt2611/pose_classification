# version cmake toi thieu can co
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(APP "pose_app")
project(${APP})

set(CMAKE_CXX_STANDARD 20)
set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS}")
set(CMAKE_PREFIX_PATH /home/kikai/projects/libs/libtorch)
set(CMAKE_BUILD_TYPE Debug)

# utils in project
set(INC ${INC} /home/kikai/projects/pose_clf_cpp/utils)

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS opencv_imgcodecs opencv_highgui)
set(LIB ${LIB} ${OpenCV_LIBRARIES})
set(INC ${INC} ${OpenCV_INCLUDE_DIRS})
message(${OpenCV_INCLUDE_DIRS})

# Torch
# Liên kết với libtorch cài bởi python
# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
# set(LIB ${LIB} ${TORCH_LIBRARIES})
# set(INC ${INC} ${TORCH_INCLUDE_DIRS})


# Liên kết với libtorch ở trong libs folder
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
set(LIB ${LIB} ${TORCH_LIBRARIES})
set(INC ${INC} ${TORCH_INCLUDE_DIRS})
message(${TORCH_LIBRARIES})
message(${TORCH_INCLUDE_DIRS})

# Darknet
set(DARKNET_PATH /home/kikai/projects/libs/darknet)
set(DARKNET_LIB_NAME libdarknet)
include_directories(${DARKNET_PATH}/include)
include_directories(${DARKNET_PATH}/src)
add_library(${DARKNET_LIB_NAME} SHARED IMPORTED)
set_property(
        TARGET ${DARKNET_LIB_NAME}
        PROPERTY IMPORTED_LOCATION 
        "${DARKNET_PATH}/build_release/libdarknet.so"
)

include_directories(${CMAKE_SOURCE_DIR}/utils)

# Main.cpp
set(SRC ${SRC} detector.cpp)
set(SRC ${SRC} utils/simple_transform.cpp)
set(SRC ${SRC} pose_estimator.cpp)
set(SRC ${SRC} main.cpp)

include_directories(${INC})
add_executable(${APP} ${SRC})
target_link_libraries(${APP} ${LIB} ${DARKNET_LIB_NAME})
