include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
    GIT_SHALLOW TRUE
    SYSTEM
)
FetchContent_MakeAvailable(benchmark)

add_executable(bench_matmul bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE mulled_ispc benchmark::benchmark benchmark::benchmark_main)
