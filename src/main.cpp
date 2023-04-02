#include "bridge.hpp"
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

template<typename T>
void print(fastllama::RingBuffer<T> const& r) {
    std::cout<<"[ ";
    for(auto i : r) {
        std::cout<<i<<", ";
    }
    std::cout<<"]"<<std::endl;
}

int main() {
    // auto bridge = fastllama::FastLlama("LLAMA-7B", "/Users/amit/Desktop/code/fastLLaMa/models/7B/ggml-model-q4_0.bin", 16, 512, 64, 0);
    // auto model = fastllama::Model("LLAMA-7B", "/Users/amit/Desktop/code/fastLLaMa/models/7B/ggml-model-q4_0.bin");
    // // llama_eval(m_model, m_threads, 0, { 0, 1, 2, 3 }, m_logits, m_mem_per_token)
    // std::vector<float> m_logits;
    // std::size_t mem_per_token{};
    // if (model.eval(8, 0, { 0, 1, 2, 3 }, m_logits, mem_per_token )) {
    //     return 1;
    // }
    
    return 0;
}