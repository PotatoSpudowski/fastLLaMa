# set(PYBIND11_FINDPYTHON ON)
# find_package(Python 3.11 EXACT COMPONENTS Interpreter Development)

# include(FetchContent)
# FetchContent_Declare(
# 	pybind11
# 	GIT_REPOSITORY https://github.com/pybind/pybind11.git
# 	GIT_TAG        v2.10.4
# 	GIT_SHALLOW    TRUE
# )

# FetchContent_MakeAvailable(pybind11)

# pybind11_add_module(pyfastllama main.cpp)

# message(STATUS "Python version built for: ${Python_VERSION}")
# target_link_libraries(pyfastllama PRIVATE fast_llama_lib)
# set_target_properties(pyfastllama PROPERTIES POSITION_INDEPENDENT_CODE ON)
include_directories(..)

add_library(pyfastllama SHARED main.cpp)
target_link_libraries(pyfastllama PRIVATE fast_llama_lib)
set_target_properties(pyfastllama PROPERTIES SUFFIX ".so")
set_target_properties(pyfastllama PROPERTIES PREFIX "")
set_target_properties(pyfastllama PROPERTIES OUTPUT_NAME "pyfastllama")
