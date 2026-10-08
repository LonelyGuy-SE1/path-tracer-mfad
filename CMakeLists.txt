cmake_minimum_required(VERSION 3.22)
project(path_tracer_mfad VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization and compiler warnings
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# OpenMP
find_package(OpenMP REQUIRED)

# Dependencies via FetchContent
include(FetchContent)

# 1. Eigen3 (Header-only Linear Algebra)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    Eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(Eigen3)

# 2. nlohmann/json (Header-only JSON parser for scene and trace)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# 3. stb (stb_image_write.h for PNG output)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(stb)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${stb_SOURCE_DIR}
)

# Project targets
# mfad_trace: trace logging library
file(GLOB_RECURSE TRACE_SOURCES "trace/*.cpp")
add_library(mfad_trace STATIC ${TRACE_SOURCES})
target_link_libraries(mfad_trace PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json)
target_include_directories(mfad_trace PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/trace)

# mfad_stages: pure linear algebra modules (Eigen only)
file(GLOB_RECURSE STAGES_SOURCES "stages/*.cpp")
add_library(mfad_stages STATIC ${STAGES_SOURCES})
target_link_libraries(mfad_stages PUBLIC Eigen3::Eigen mfad_trace)
target_include_directories(mfad_stages PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/stages
    ${CMAKE_CURRENT_SOURCE_DIR}/trace
)

# mfad_core: renderer core library
file(GLOB_RECURSE CORE_SOURCES "core/*.cpp")
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*main\\.cpp$")
if(NOT CORE_SOURCES)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/core_dummy.cpp" "// dummy\n")
    set(CORE_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/core_dummy.cpp")
endif()
add_library(mfad_core STATIC ${CORE_SOURCES})
target_link_libraries(mfad_core PUBLIC
    mfad_stages
    mfad_trace
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    nlohmann_json::nlohmann_json
)
target_include_directories(mfad_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core
)
target_include_directories(mfad_core SYSTEM PUBLIC
    ${stb_SOURCE_DIR}
)

# Main executable: path_tracer
add_executable(path_tracer core/main.cpp)
target_link_libraries(path_tracer PRIVATE
    mfad_core
    mfad_stages
    mfad_trace
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# Testing configuration
enable_testing()

# Test executable for stage_basis
add_executable(test_basis_runner tests/test_basis.cpp)
target_link_libraries(test_basis_runner PRIVATE
    mfad_stages
    mfad_trace
    Eigen3::Eigen
)
target_include_directories(test_basis_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stages
    ${CMAKE_CURRENT_SOURCE_DIR}/trace
)
add_test(NAME test_basis COMMAND test_basis_runner)

# Test executable for intersections
add_executable(test_intersections_runner tests/test_intersections.cpp)
target_link_libraries(test_intersections_runner PRIVATE
    mfad_core
    Eigen3::Eigen
)
target_include_directories(test_intersections_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
)
add_test(NAME test_intersections COMMAND test_intersections_runner)

# Test executable for materials (Issue #25)
add_executable(test_materials_runner tests/test_materials.cpp)
target_link_libraries(test_materials_runner PRIVATE
    mfad_core
    mfad_stages
    mfad_trace
    Eigen3::Eigen
)
target_include_directories(test_materials_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/stages
    ${CMAKE_CURRENT_SOURCE_DIR}/trace
)
add_test(NAME test_materials COMMAND test_materials_runner)

# Test executable for path tracer & Russian roulette (Issue #26)
add_executable(test_path_tracer_runner tests/test_path_tracer.cpp)
target_link_libraries(test_path_tracer_runner PRIVATE
    mfad_core
    mfad_stages
    mfad_trace
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)
target_include_directories(test_path_tracer_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/stages
    ${CMAKE_CURRENT_SOURCE_DIR}/trace
)
add_test(NAME test_path_tracer COMMAND test_path_tracer_runner)
