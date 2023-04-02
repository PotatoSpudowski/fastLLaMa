cmake_minimum_required(VERSION 3.20)
project(fastLlama)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Release)

include(cmake/CompilerFlags.cmake)
include(cmake/OpenMP.cmake)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(fast_llama_lib ${CMAKE_CURRENT_SOURCE_DIR}/lib/ggml.c ${CMAKE_CURRENT_SOURCE_DIR}/lib/utils.cpp ${CMAKE_CURRENT_SOURCE_DIR}/lib/llama.cpp ${CMAKE_CURRENT_SOURCE_DIR}/lib/bridge.cpp)

set_compiler_lib_and_flags(fast_llama_lib)
target_link_libraries(fast_llama_lib PRIVATE ${OPENMP_LIB})
# target_compile_options(fast_llama_lib PRIVATE "-g")
set_target_properties(fast_llama_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.10.4
	GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(pybind11)

# pybind11_add_module(fastLlama bridge.cpp)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# link_directories(${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(fastLlama PRIVATE fast_llama_lib)
# set_target_properties(fastLlama PROPERTIES POSITION_INDEPENDENT_CODE ON)
# set_target_properties(fastLlama PROPERTIES SUFFIX ".so")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
