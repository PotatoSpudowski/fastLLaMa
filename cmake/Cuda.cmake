set(_CMAKE_CUDA_WHOLE_FLAG "-c")

if(NOT CMAKE_CUDA_COMPILE_OBJECT)
  set(CMAKE_CUDA_COMPILE_OBJECT
    "<CMAKE_CUDA_COMPILER> ${_CMAKE_CUDA_EXTRA_FLAGS} <DEFINES> <INCLUDES> <FLAGS> ${_CMAKE_COMPILE_AS_CUDA_FLAG} <CUDA_COMPILE_MODE> <SOURCE> -o <OBJECT>")
endif()

option(LLAMA_CUBLAS "fastllama: use cuBLAS" ON)

function(set_cuda project_name)

    if(LLAMA_CUBLAS)
        cmake_minimum_required(VERSION 3.17)

        find_package(CUDAToolkit)
        
        if (CUDAToolkit_FOUND)
            message(STATUS "cuBLAS found")

            enable_language(CUDA)

            target_sources(${project_name} PRIVATE ${CMAKE_SOURCE_DIR}/lib/ggml-cuda.cu ${CMAKE_SOURCE_DIR}/include/ggml-cuda.h)
            target_link_libraries(${project_name} PRIVATE CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
            target_compile_definitions(${project_name} PRIVATE "GGML_USE_CUBLAS")

        endif()
    endif(LLAMA_CUBLAS)
    
    
endfunction(set_cuda project_name)

