include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BasicUtils.cmake)

set(COMPILER_FLAG_VARS_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompilerFlagVariables.cmake")
set(COMPILER_CXX_FLAG "")

if (EXISTS ${COMPILER_FLAG_VARS_FILE_PATH})
    include(${COMPILER_FLAG_VARS_FILE_PATH})
    _COMPILER_SWITCH_SET(COMPILER_CXX_FLAG ${GCC_CXXFLAG} ${CLANG_CXXFLAG} ${MSVC_CXXFLAG})
else()
    _COMPILER_SWITCH_SET(COMPILER_CXX_FLAG "-march=native" "-march=native" "")
endif(EXISTS ${COMPILER_FLAG_VARS_FILE_PATH})


function(set_compiler_lib_and_flags project_name)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)

    set(COMPILER_FLAGS_LIST ${COMPILER_CXX_FLAG})
    set(COMPILER_LDFLAGS_LIST "")
    set(COMPILER_DEF_FLAGS_LIST "")

    if(Threads_FOUND)
        target_link_libraries(${project_name} PRIVATE Threads::Threads)
    endif(Threads_FOUND)
    
    
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        if(NOT DEFINED LLAMA_NO_ACCELERATE)
            list(APPEND COMPILER_DEF_FLAGS_LIST "GGML_USE_ACCELERATE")
            _COMPILER_SWITCH_APPEND(COMPILER_LDFLAGS_LIST "-framework Accelerate" "-framework Accelerate" "")
        endif(NOT DEFINED LLAMA_NO_ACCELERATE)
    endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    
    if(${CMAKE_SYSTEM_NAME} MATCHES "aarch64")
        _COMPILER_SWITCH_APPEND(COMPILER_FLAGS_LIST "-mcpu=native" "-mcpu=native" "")
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "armv6")
        set(_ARM_FLAG "-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access")
        _COMPILER_SWITCH_APPEND(COMPILER_FLAGS_LIST ${_ARM_FLAG} ${_ARM_FLAG} "")
        unset(_ARM_FLAG)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "armv7")
        set(_ARM_FLAG "-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations")
        _COMPILER_SWITCH_APPEND(COMPILER_FLAGS_LIST ${_ARM_FLAG} ${_ARM_FLAG} "")
        unset(_ARM_FLAG)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "armv8")
        set(_ARM_FLAG "-mfp16-format=ieee -mno-unaligned-access")
        _COMPILER_SWITCH_APPEND(COMPILER_FLAGS_LIST ${_ARM_FLAG} ${_ARM_FLAG} "")
        unset(_ARM_FLAG)
    endif()
    
    string(REPLACE ";" " " COMPILER_FLAGS "${COMPILER_FLAGS_LIST}")
    string(REPLACE ";" " " COMPILER_LDFLAGS "${COMPILER_LDFLAGS_LIST}")
    string(REPLACE ";" " " COMPILER_DEF_FLAGS "${COMPILER_DEF_FLAGS_LIST}")
    
    target_compile_options(${project_name} PRIVATE ${COMPILER_FLAGS})
    target_link_options(${project_name} PRIVATE ${COMPILER_LDFLAGS})
    target_compile_definitions(${project_name} PRIVATE ${COMPILER_DEF_FLAGS})

endfunction(set_compiler_lib_and_flags project_name)
