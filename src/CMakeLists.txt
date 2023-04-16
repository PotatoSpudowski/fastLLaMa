add_executable(quantize quantize.cpp)
target_link_libraries(quantize PRIVATE fast_llama_lib)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE fast_llama_lib)
target_compile_options(main PRIVATE -O0 -g)

add_executable(alpaca alpaca.cpp)
target_link_libraries(alpaca PRIVATE fast_llama_lib)

# target_compile_options(main PRIVATE "-g")