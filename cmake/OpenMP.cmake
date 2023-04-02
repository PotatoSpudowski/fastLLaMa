option(ENABLE_OPENMP "Enable OpenMP" ON)
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found")
        set(OPENMP_LIB OpenMP::OpenMP_CXX)
    else()
        message(FATAL_ERROR "OpenMP not found")
    endif()
else()
    set(OPENMP_LIB)
endif(ENABLE_OPENMP)
