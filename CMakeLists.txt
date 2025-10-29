cmake_minimum_required(VERSION 3.16)
project(lidar_odometry LANGUAGES CXX)

# Tiziano Says:
# Do not set global configurations -> Target oriented CMake for the win
# I think its best to combine both

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# generates .json file with all compile commands (like if were writting without CMake)

set(CMAKE_CXX_STANDARD 17)

# for all targets (libraries and executables)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#include_directories(include/) # include_directoris for all targets (i.e. all libraries and executables)

#include_directories(/usr/include) # for system-wide installed libraries. below approach is the same, but considered more modern
#add_library(global_includes INTERFACE)
#target_include_directories(global_includes INTERFACE /usr/include)

#### Pre-compiled libraries
find_package(yaml-cpp REQUIRED)

# headers: /usr/include/yaml-cpp/
find_package(Eigen3 3.2 REQUIRED)

# headers: /usr/include/eigen3/

#### Library
# 1. Create library "libconfig"
add_library(config STATIC utils/config.cpp)
# generates libconfig.a (STATIC is by default) ; SHARED -> .so library
# 2. Define C++ standard for this library
# target_compile_features(config PRIVATE cxx_std_11)
# 3. Tell CMake where to look for .h or .hpp files for this specific library (i.e. target)
target_link_libraries(config PRIVATE yaml-cpp)

# when I link my library to another library, its include directories are automatically added for my library, so I do not need to specify them manually
target_include_directories(config PUBLIC include/)
#target_link_libraries(config PRIVATE global_includes)

add_library(dataset STATIC utils/dataset.cpp)
target_include_directories(dataset PUBLIC include/)
target_link_libraries(dataset PUBLIC Eigen3::Eigen)

#### Executable
add_executable(main main.cpp)

#### Linking ("keeping the promises from Preprocessing step of compilation")
target_link_libraries(main PRIVATE config dataset)

#### Printing
get_target_property(DATASET_INCS dataset INCLUDE_DIRECTORIES)
get_target_property(CONFIG_INCS config INCLUDE_DIRECTORIES)
get_target_property(MAIN_INCS main INCLUDE_DIRECTORIES)

message(STATUS "Include directories for Eigen: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Include directories for config: ${CONFIG_INCS}")
message(STATUS "Include directories for dataset: ${DATASET_INCS}")
message(STATUS "Include directories for yaml-cpp: ${YAML_CPP_INCLUDE_DIR}")
message(STATUS "Include directories for main: ${MAIN_INCS}")
