option(LLAMA_CUBLAS "fastllama: use cuBLAS" ON)

function(set_cuda project_name)

    if(LLAMA_CUBLAS)
        find_package(CUDAToolkit)
        if (CUDAToolkit_FOUND)
            message(STATUS "cuBLAS found")

            enable_language(CUDA)

            target_link_libraries(${project_name} PRIVATE ${CMAKE_SOURCE_DIR}/ggml-cuda.cu ${CMAKE_SOURCE_DIR}/ggml-cuda.h CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
            target_compile_definitions(${project_name} PRIVATE "GGML_USE_CUBLAS")

        endif()
    endif(LLAMA_CUBLAS)
    
    
endfunction(set_cuda project_name)

