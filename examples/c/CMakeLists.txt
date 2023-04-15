cmake_minimum_required(VERSION 3.20)
project(fastLlama)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(GLOBAL_VARS_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/GlobalVars.cmake")

if(EXISTS ${GLOBAL_VARS_FILE_PATH})
  message(STATUS "Found '${GLOBAL_VARS_FILE_PATH}' for 'C'")
  include(${GLOBAL_VARS_FILE_PATH})

  include_directories(${PYTHON_INCLUDE_DIRS})
  link_libraries(${PYTHON_LIBRARIES})
else()
  message(STATUS "Unable to find '${GLOBAL_VARS_FILE_PATH}' for 'C'")
endif(EXISTS ${GLOBAL_VARS_FILE_PATH})

include(${WORKSPACE}/cmake/OpenMP.cmake)
include_directories(${WORKSPACE}/interfaces/c)
link_directories(${WORKSPACE}/build/interfaces/c)
link_directories(${WORKSPACE}/build)
include(${WORKSPACE}/cmake/OpenBLAS.cmake)

macro(_SET_OPTIONS_AND_LIB target)
  target_link_libraries(${target} PRIVATE cfastllama)
  target_link_libraries(${target} PRIVATE fast_llama_lib)
  target_link_libraries(${target} PRIVATE ggml_library)

  target_link_libraries(${target} PRIVATE ${OPENMP_LIB_C})

  if (NOT WIN32)
    target_link_libraries(${target} PRIVATE stdc++)
    target_link_libraries(${target} PRIVATE m)
  endif(NOT WIN32)

  if(DEFINED LLAMA_NO_ACCELERATE OR DEFINED LLAMA_OPENBLAS)
      set_open_blas(${target})
  else()
      if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
          target_compile_definitions(${target} PRIVATE "GGML_USE_ACCELERATE")
          target_link_libraries(${target} PRIVATE "-framework Accelerate")
      endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  endif(DEFINED LLAMA_NO_ACCELERATE OR DEFINED LLAMA_OPENBLAS)
endmacro(_SET_OPTIONS_AND_LIB target)

add_executable(example example.c)
_SET_OPTIONS_AND_LIB(example)

add_executable(example-alpaca example-alpaca.c)
_SET_OPTIONS_AND_LIB(example-alpaca)

add_executable(perplexity perplexity.c)
_SET_OPTIONS_AND_LIB(perplexity)
