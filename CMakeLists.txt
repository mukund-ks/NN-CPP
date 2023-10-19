# Starter CMakeLists file
cmake_minimum_required(VERSION 3.22)
project(NN-CPP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

include_directories(include)

set(SOURCE_DIR src)

add_executable(NN-CPP
    ${SOURCE_DIR}/activation_fn.cpp
    ${SOURCE_DIR}/data_loader.cpp
    ${SOURCE_DIR}/layers.cpp
    ${SOURCE_DIR}/loss_fn.cpp
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/neural_networks.cpp
    ${SOURCE_DIR}/optimizers.cpp
)