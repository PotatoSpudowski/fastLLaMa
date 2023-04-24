function(set_cuda project_name)
    find_package(CUDAToolkit QUIET)
    if(BLAS_FOUND)
        target_link_libraries(${project_name} PRIVATE CUDA::cublas)
        target_compile_definitions(${project_name} PRIVATE "GGML_USE_CUBLAS")
    endif(BLAS_FOUND)
    
endfunction(set_cuda project_name)

