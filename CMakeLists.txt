cmake_minimum_required(VERSION 3.20)
project(fastLlama)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Release)

include(cmake/CompilerFlags.cmake)

add_library(fast_llama_lib ${CMAKE_CURRENT_SOURCE_DIR}/ggml.c ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)

set_compiler_lib_and_flags(fast_llama_lib)
set_target_properties(fast_llama_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(quantize ${CMAKE_CURRENT_SOURCE_DIR}/quantize.cpp)
target_link_libraries(quantize PRIVATE fast_llama_lib)

include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.10.4
	GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(fastLlama bridge.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${CMAKE_CURRENT_SOURCE_DIR})
# set(LLAMA_NO_ACCELERATE 1)
# set_compiler_lib_and_flags(fastLlama)
target_link_libraries(fastLlama PRIVATE fast_llama_lib)
set_target_properties(fastLlama PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(fastLlama PROPERTIES SUFFIX ".so")
