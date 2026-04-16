include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.3.2
    GIT_SHALLOW TRUE
    SYSTEM
)
FetchContent_MakeAvailable(Catch2)

include(CTest)
include(Catch)

add_executable(test_runner test_matmul.cpp)
target_link_libraries(test_runner PRIVATE mulled_ispc Catch2::Catch2WithMain)

catch_discover_tests(test_runner)
