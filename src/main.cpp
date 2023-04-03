#include "bridge.hpp"
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

<<<<<<< HEAD
int main() {
    auto maybe_bridge = fastllama::FastLlama::Builder()
        .set_number_of_threads(16)
        .build("LLAMA-7B", "/Users/amit/Desktop/code/fastLLaMa/models/7B/ggml-model-q4_0.bin");
    
    if (!maybe_bridge) {
        return 1;
    }
    auto bridge = maybe_bridge.value();
    // auto bridge = fastllama::FastLlama("LLAMA-7B", "/Users/amit/Desktop/code/fastLLaMa/models/7B/ggml-model-q4_0.bin", 16);
    // bridge.dump_vocab("./vocab.txt");
    auto& logger = bridge.get_logger();

    logger.log_warn("main", "Ingesting, please wait...\n");
    bridge.ingest(
        "Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.\n"
        "User: Hello, Bob.\n"
        "Bob: Hello. How may I help you today?\n"
        "User: Please tell me the largest city in Europe.\n"
        "Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.\n"
        "User:\n"
    );

    logger.log_warn("main", "Ingestion complete!\n");

    bridge.generate([](std::string const& s) {
        std::cout<<s;
        std::cout.flush();
    }, 300, 40, 0.95, 0.8, 1.0);

=======
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
>>>>>>> 411b2fe3005bd12b4d974594b5d40573f7625a22
    // auto model = fastllama::Model("LLAMA-7B", "/Users/amit/Desktop/code/fastLLaMa/models/7B/ggml-model-q4_0.bin");
    // // llama_eval(m_model, m_threads, 0, { 0, 1, 2, 3 }, m_logits, m_mem_per_token)
    // std::vector<float> m_logits;
    // std::size_t mem_per_token{};
    // if (model.eval(8, 0, { 0, 1, 2, 3 }, m_logits, mem_per_token )) {
    //     return 1;
    // }
    
    return 0;
}