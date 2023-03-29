macro(_SET_FLAG var gcc clang msvc)
    if(MSVC)
        list(APPEND ${var} ${msvc})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
        list(APPEND ${var} ${clang})
    else()
        list(APPEND ${var} ${gcc})
    endif()
endmacro(_SET_FLAG)

function(set_compiler_lib_and_flags link_target)
    set(x86_ARCH_REGEX "x86_64|amd64|AMD64|x86|i386|i686")

    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)

    set(COMPILER_FLAGS)

    if(Threads_FOUND)
        target_link_libraries(${link_target} INTERFACE Threads::Threads)
    endif(Threads_FOUND)
        
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES ${x86_ARCH_REGEX})    
        _SET_FLAG(COMPILER_FLAGS "" "" "/GL")

        if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
            _SET_FLAG(COMPILER_FLAGS "-mf16c" "-mf16c" "")
        endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    endif(${CMAKE_SYSTEM_PROCESSOR} MATCHES ${x86_ARCH_REGEX})
        
        
    
    message(STATUS "Amit: ${COMPILER_FLAGS}")

    # target_compile_options(${link_target} PRIVATE "-std=c++17")

endfunction(set_compiler_lib_and_flags link_target)
