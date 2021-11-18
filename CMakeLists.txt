cmake_minimum_required(VERSION 3.0)

project(MapEvaluationTools)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(pybind11)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable (test src/test.cpp src/map_evaluation.hpp)
target_link_libraries (test ${PCL_LIBRARIES})

#add_executable (orth_map_metrics src/orth_map_metrics.cpp)
#target_link_libraries (orth_map_metrics ${PCL_LIBRARIES})

# pybind11
pybind11_add_module(map_evaluation src/map_evaluation.hpp src/map_evaluation.cpp)
set_target_properties(map_evaluation PROPERTIES PREFIX "")
set_target_properties(map_evaluation PROPERTIES SUFFIX ".so")
target_link_libraries(map_evaluation PRIVATE ${PYTHON_LIBARIES} ${PCL_LIBRARIES})

