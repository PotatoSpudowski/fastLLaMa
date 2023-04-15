add_library(cfastllama STATIC main.cpp)
target_link_libraries(cfastllama PRIVATE fast_llama_lib)
