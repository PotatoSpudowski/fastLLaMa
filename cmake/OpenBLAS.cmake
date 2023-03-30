function(set_open_blas project_name)
    set(BLA_VENDOR "OpenBLAS")
    find_package(BLAS)
    if(BLAS_FOUND)
        target_link_libraries(${project_name} PRIVATE BLAS::BLAS)
        target_compile_definitions(${project_name} PRIVATE "GGML_USE_OPENBLAS")
    endif(BLAS_FOUND)
    
endfunction(set_open_blas project_name)

