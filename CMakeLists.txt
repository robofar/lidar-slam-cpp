cmake_minimum_required(VERSION 3.16)
project(lidar_odometry LANGUAGES CXX)

# Tiziano Says:
# Do not set global configurations -> Target oriented CMake for the win
# I think its best to combine both

set(CMAKE_BUILD_TYPE RelWithDebInfo)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# generates .json file with all compile commands (like if were writting without CMake)

set(CMAKE_CXX_STANDARD 17)

# for all targets (libraries and executables)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#include_directories(include/) # include_directoris for all targets (i.e. all libraries and executables)

#include_directories(/usr/include) # for system-wide installed libraries. below approach is the same, but considered more modern
#add_library(global_includes INTERFACE)
#target_include_directories(global_includes INTERFACE /usr/include)

# headers: /usr/include/yaml-cpp/
#find_package(Eigen3 3.3.0 REQUIRED)

# Cloned and checked-out tag 3.4.1
add_subdirectory(./3rdparty/eigen)

set(SOPHUS_USE_BASIC_LOGGING ON CACHE BOOL "Don't use fmt for Sophus library")
set(BUILD_SOPHUS_TESTS OFF CACHE BOOL "Don't build Sophus tests")
set(BUILD_SOPHUS_EXAMPLES OFF CACHE BOOL "Don't build Sophus Examples")
add_subdirectory(./3rdparty/Sophus)

#### Pre-compiled libraries
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)

# headers: /usr/include/eigen3/

#### Library

# 1. Create library "libconfig"
add_library(config STATIC utils/config.cpp)
# generates libconfig.a (STATIC is by default) ; SHARED -> .so library
# 2. Define C++ standard for this library
# target_compile_features(config PRIVATE cxx_std_11)
# 3. Tell CMake where to look for .h or .hpp files for this specific library (i.e. target)
target_link_libraries(config PRIVATE yaml-cpp)

# when I link my library to another library, its include directories are automatically added for my library (if PUBLIC), so I do not need to specify them manually
target_include_directories(config PRIVATE include/)
#target_link_libraries(config PRIVATE global_includes)

add_library(kitti STATIC dataloaders/kitti.cpp)
target_include_directories(kitti PUBLIC include/ include/dataloaders/)
target_link_libraries(kitti PUBLIC config PUBLIC Eigen3::Eigen PUBLIC Sophus::Sophus)

add_library(dataset STATIC utils/dataset.cpp)
target_include_directories(dataset PUBLIC include/ PUBLIC 3rdparty/Sophus/sophus/)
target_link_libraries(dataset PUBLIC config PUBLIC Eigen3::Eigen PUBLIC Sophus::Sophus PUBLIC kitti)

add_library(vhm STATIC utils/vhm.cpp)
# bcs it links to config, it automatically inherits its directories path (i.e. include/, where vhm.hpp is also located)
target_include_directories(vhm PUBLIC include/)
target_link_libraries(vhm PUBLIC config PUBLIC Eigen3::Eigen PUBLIC Sophus::Sophus PUBLIC OpenMP::OpenMP_CXX)

add_library(mapper STATIC utils/mapper.cpp)
target_include_directories(mapper PUBLIC include/)
target_link_libraries(mapper PUBLIC config PUBLIC dataset PUBLIC vhm PUBLIC Eigen3::Eigen)

add_library(tracker STATIC utils/tracker.cpp)
target_include_directories(tracker PUBLIC include/)
target_link_libraries(tracker PUBLIC config PUBLIC vhm PUBLIC Eigen3::Eigen PUBLIC Sophus::Sophus)

# Download the rerun_sdk
include(FetchContent)
FetchContent_Declare(rerun_sdk URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

add_library(visualizer STATIC utils/visualizer.cpp)
target_include_directories(visualizer PUBLIC include/)
# Link visualizer against rerun_sdk
target_link_libraries(visualizer PUBLIC config PUBLIC Eigen3::Eigen PUBLIC Sophus::Sophus PUBLIC rerun_sdk)

#### Executable
add_executable(main main.cpp)

#### Linking ("keeping the promises from Preprocessing step of compilation")
#target_link_libraries(main PRIVATE config dataset)
target_link_libraries(
  main
  PRIVATE kitti
  PRIVATE dataset
  PRIVATE config
  PRIVATE vhm
  PRIVATE mapper
  PRIVATE tracker
  PUBLIC visualizer)
