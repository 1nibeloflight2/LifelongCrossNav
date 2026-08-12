cmake_minimum_required(VERSION 3.27)
project(path)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_BUILD_TYPE Release)
find_package (Python COMPONENTS Interpreter Development)

pybind11_add_module(planning_utils_cpp djikstra.cpp matrix_utils.h)

target_link_libraries(planning_utils_cpp PRIVATE Eigen3::Eigen)

target_include_directories(planning_utils_cpp PRIVATE
        ${PYBIND11_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)
install(TARGETS planning_utils_cpp LIBRARY DESTINATION .)



#include(FetchContent)
#FetchContent_Declare(rerun_sdk URL
#        https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
#FetchContent_MakeAvailable(rerun_sdk)
#
#target_link_libraries(path PRIVATE rerun_sdk)

