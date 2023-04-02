#include "llama.hpp"

int main() {

    auto model = fastllama::Model("LLAMA-7B", "/Users/amit/Desktop/code/fastLLaMa/models/7B/ggml-model-q4_0.bin", 512);
}