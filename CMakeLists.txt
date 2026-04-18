cmake_minimum_required(VERSION 3.31)

project(mulled
    VERSION 0.1.0
    DESCRIPTION "Matrix multiplication implementations and benchmarks"
    LANGUAGES CXX ISPC
)

message(STATUS "Configuring mulled version: ${PROJECT_VERSION}")

include(FetchContent)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.0
    GIT_SHALLOW TRUE
    SYSTEM
)
FetchContent_MakeAvailable(Eigen3)

FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG mdspan-0.6.0
    GIT_SHALLOW TRUE
    SYSTEM
)
FetchContent_MakeAvailable(mdspan)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED IMPORTED_TARGET openblas)

add_library(mulled INTERFACE)
target_include_directories(mulled INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(mulled INTERFACE
    $<$<CONFIG:Release>:-march=native -ffast-math>
    -fdiagnostics-color=always
    -Werror -Wall -Wextra -Wpedantic -Wfloat-conversion -Wshadow -Wdouble-promotion
)
target_compile_features(mulled INTERFACE cxx_std_26)
target_link_libraries(mulled INTERFACE Eigen3::Eigen PkgConfig::OPENBLAS std::mdspan)

add_library(mulled_ispc STATIC src/matmul.ispc)
target_compile_options(mulled_ispc PRIVATE
    $<$<CONFIG:Release>:--opt=fast-math>
)
target_include_directories(mulled_ispc INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/mulled_ispc.dir>)
target_link_libraries(mulled_ispc INTERFACE mulled)

enable_testing()

add_subdirectory(tests)
add_subdirectory(benchmarks)
