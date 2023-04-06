include_directories(${CMAKE_SOURCE_DIR}/interfaces/c)

add_executable(example example.c)
target_link_libraries(example PRIVATE cfastllama)
