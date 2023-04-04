#include "bridge.hpp"
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

int main() {
    auto maybe_bridge = fastllama::FastLlama::builder()
        .set_number_of_threads(16)
        .set_number_of_batches(64)
        .set_number_of_tokens_to_keep(48)
        .build("LLAMA-7B", "./models/7B/ggml-model-q4_0.bin");
    
    if (!maybe_bridge) {
        return 1;
    }
    auto bridge = maybe_bridge.value();
    // bridge.dump_vocab("./vocab.txt");
    auto& logger = bridge.get_logger();

    logger.log_warn("main", "Ingesting, please wait...\n");
    bridge.ingest(
        "Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.\n"
        "User: Hello, Bob.\n"
        "Bob: Hello. How may I help you today?\n"
        "User: Please tell me the largest city in Europe.\n"
        "Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.\n"
    );
    logger.log_warn("main", "Ingestion complete!\n");

    std::string prompt;

    std::cout<<"User: ";

    while(std::getline(std::cin, prompt)) {
        prompt = "User: " + prompt;
        
        bridge.ingest(prompt);

        bridge.generate([](std::string const& s) {
            std::cout<<s;
            std::cout.flush();
        }, 300, 40, 0.95, 0.8, 1.0, { "User: " });
        
        std::cout<<"User: ";
    }
    
    return 0;
}