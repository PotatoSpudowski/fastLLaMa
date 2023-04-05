cmake_minimum_required(VERSION 3.20)
project(fastLlama)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(cmake/CompilerWarnings.cmake)
include(cmake/CompilerFlags.cmake)
include(cmake/OpenMP.cmake)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(fast_llama_lib ${CMAKE_CURRENT_SOURCE_DIR}/lib/ggml.c ${CMAKE_CURRENT_SOURCE_DIR}/lib/llama.cpp ${CMAKE_CURRENT_SOURCE_DIR}/lib/bridge.cpp)
set_project_warnings(fast_llama_lib)

set_compiler_lib_and_flags(fast_llama_lib)
target_link_libraries(fast_llama_lib PRIVATE ${OPENMP_LIB})
# target_compile_options(fast_llama_lib PRIVATE "-g")
set_target_properties(fast_llama_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/interfaces)
