# Starter CMakeLists file
cmake_minimum_required(VERSION 3.22)
project(NN-CPP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(SOURCE_DIR src)
set(INCLUDE_DIR include)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})

set(SOURCE_FILES 
    ${SOURCE_DIR}/activation_fn.cpp 
    ${SOURCE_DIR}/data_loader.cpp 
    ${SOURCE_DIR}/layers.cpp 
    ${SOURCE_DIR}/loss_fn.cpp 
    ${SOURCE_DIR}/neural_network.cpp 
    ${SOURCE_DIR}/optimizers.cpp 
    ${SOURCE_DIR}/main.cpp
)

add_executable(NN-CPP ${SOURCE_FILES})

target_link_libraries(NN-CPP ${OpenCV_LIBS})
target_link_libraries(NN-CPP ${Eigen3_LIBS})