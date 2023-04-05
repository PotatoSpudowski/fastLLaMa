find_package(Python ${PYTHON_VERSION} COMPONENTS Interpreter Development)

include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.10.4
	GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(fastLlama main.cpp)

message(STATUS "Python version built for: ${Python_VERSION}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fastLlama PRIVATE fast_llama_lib)
set_target_properties(fastLlama PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(fastLlama PROPERTIES SUFFIX ".so")
