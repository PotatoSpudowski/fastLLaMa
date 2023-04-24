include(cmake/BasicUtils.cmake)
include(CheckCXXCompilerFlag)
include(cmake/OpenBLAS.cmake)

set(COMPILER_FLAG_VARS_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompilerFlagVariables.cmake")
set(COMPILER_CXX_FLAG "")

macro(_NORMALIZE_COMPILER_FLAGS VAR FLAGS)
    set(_INTERNAL_VAR "")
    foreach(FLAG ${FLAGS})
        string(REGEX REPLACE "(-)|(/arch:)|/" "" FLAG_NAME ${FLAG})
        check_cxx_compiler_flag(${FLAG} C_FLAG_SUPPORTED_${FLAG_NAME})
        if(C_FLAG_SUPPORTED_${FLAG_NAME})
            list(APPEND _INTERNAL_VAR "${FLAG}")
        endif(C_FLAG_SUPPORTED_${FLAG_NAME})
        unset(C_FLAG_SUPPORTED_${FLAG_NAME})
    endforeach(FLAG ${FLAGS})
    set(${VAR} ${_INTERNAL_VAR})
    unset(_INTERNAL_VAR)
endmacro(_NORMALIZE_COMPILER_FLAGS var flags)

check_cxx_compiler_flag(-march=native NATIVE_C_FLAG_SUPPORTED)

if (EXISTS ${COMPILER_FLAG_VARS_FILE_PATH})
    include(${COMPILER_FLAG_VARS_FILE_PATH})
    if(NATIVE_C_FLAG_SUPPORTED)
        _COMPILER_SWITCH_SET(COMPILER_CXX_FLAG "-march=native" "-march=native" "")
    endif(NATIVE_C_FLAG_SUPPORTED)
    
    if(MSVC)
        _NORMALIZE_COMPILER_FLAGS(NORMALIZED_MSVC_FLAGS "${MSVC_CXXFLAG}")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
        _NORMALIZE_COMPILER_FLAGS(NORMALIZED_CLANG_FLAGS "${CLANG_CXXFLAG}")
    elseif(CMAKE_CUDA_HOST_COMPILER AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "${CMAKE_CUDA_HOST_COMPILER_ID}")
        # DO NOTHING
    else()
        _NORMALIZE_COMPILER_FLAGS(NORMALIZED_GCC_FLAGS "${GCC_CXXFLAG}")
    endif()
    
    _COMPILER_SWITCH_SET(COMPILER_CXX_FLAG "${NORMALIZED_GCC_FLAGS}" "${NORMALIZED_CLANG_FLAGS}" "${NORMALIZED_MSVC_FLAGS}")
else()
    if(NATIVE_C_FLAG_SUPPORTED)
        _COMPILER_SWITCH_SET(COMPILER_CXX_FLAG "-march=native" "-march=native" "")
    endif(NATIVE_C_FLAG_SUPPORTED)
endif(EXISTS ${COMPILER_FLAG_VARS_FILE_PATH})


function(set_compiler_lib_and_flags project_name compiled_lang)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)

    set(COMPILER_FLAGS_LIST ${COMPILER_CXX_FLAG})
    set(COMPILER_LDFLAGS_LIST "")
    set(COMPILER_DEF_FLAGS_LIST "")

    if(Threads_FOUND)
        target_link_libraries(${project_name} PRIVATE Threads::Threads)
    endif(Threads_FOUND)
    
    if(DEFINED LLAMA_NO_ACCELERATE OR DEFINED LLAMA_OPENBLAS)
        set_open_blas(${project_name})
    else()
        if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
            target_compile_definitions(${project_name} PRIVATE "GGML_USE_ACCELERATE")
            target_link_libraries(${project_name} PRIVATE "-framework Accelerate")
        endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    endif(DEFINED LLAMA_NO_ACCELERATE OR DEFINED LLAMA_OPENBLAS)
    
    if(${CMAKE_SYSTEM_NAME} MATCHES "aarch64")
        _COMPILER_SWITCH_APPEND(COMPILER_FLAGS_LIST "-mcpu=native" "-mcpu=native" "")
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "armv6")
        set(_ARM_FLAG "-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access")
        _COMPILER_SWITCH_APPEND(COMPILER_FLAGS_LIST "${_ARM_FLAG}" "${_ARM_FLAG}" "")
        unset(_ARM_FLAG)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "armv7")
        set(_ARM_FLAG "-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations")
        _COMPILER_SWITCH_APPEND(COMPILER_FLAGS_LIST "${_ARM_FLAG}" "${_ARM_FLAG}" "")
        unset(_ARM_FLAG)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "armv8")
        set(_ARM_FLAG "-mfp16-format=ieee -mno-unaligned-access")
        _COMPILER_SWITCH_APPEND(COMPILER_FLAGS_LIST "${_ARM_FLAG}" "${_ARM_FLAG}" "")
        unset(_ARM_FLAG)
    endif()

    if (NOT WIN32)
        target_link_libraries(${project_name} PRIVATE m)
    endif(NOT WIN32)

    if(${compiled_lang} STREQUAL "CXX")    
        
        if(NO_EXCEPTION EQUAL TRUE)
            _COMPILER_SWITCH_APPEND(COMPILER_FLAGS_LIST "-fno-exceptions" "-fno-exceptions" "/EHs-c-")
        endif(NO_EXCEPTION EQUAL TRUE)
        
        _COMPILER_SWITCH_APPEND(COMPILER_FLAGS_LIST "-fno-rtti" "-fno-rtti" "/GR-")
        
    endif()

    message(STATUS "Compiler flags used: ${COMPILER_FLAGS_LIST}")
    message(STATUS "Linking flags used: ${COMPILER_LDFLAGS_LIST}")
    message(STATUS "Macros defined: ${COMPILER_DEF_FLAGS_LIST}")

    target_compile_options(${project_name} PRIVATE ${COMPILER_FLAGS_LIST})
    target_link_options(${project_name} PRIVATE ${COMPILER_LDFLAGS_LIST})
    target_compile_definitions(${project_name} PRIVATE ${COMPILER_DEF_FLAGS_LIST})

endfunction(set_compiler_lib_and_flags project_name)