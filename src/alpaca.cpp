#include "bridge.hpp"
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

int main() {
    auto maybe_bridge = fastllama::FastLlama::builder()
        .set_number_of_threads(16)
        .set_number_of_batches(64)
        .set_number_of_contexts(512)
        .set_number_of_tokens_to_keep(48)
        .set_use_mmap(false)
        .set_use_parallel_loading(true)
        .build("./models/alpaca-lora-7B/alpaca-model");
    
    if (!maybe_bridge) {
        return 1;
    }
    auto bridge = std::move(maybe_bridge.value());
    // bridge.dump_vocab("./vocab.txt");
    auto& logger = bridge.get_logger();


    std::cout<<"\nStart of chat (type 'exit' to exit)\n";

    std::string prompt;

    std::cout<<"User: ";

    while(std::getline(std::cin, prompt)) {
        if (prompt == "exit") break;

        prompt = "### Instruction:\n\n" + prompt + "\n\n ### Response:\n\n";
        
        if (!bridge.ingest(prompt)) return 2;

        auto gen_res = bridge.generate([](std::string const& s) {
            std::cout<<s;
            std::cout.flush();
        }, 300, 40, 0.95f, 0.8f, 1.0f, { "User: " });

        if (!gen_res) return 3;
        
        std::cout<<"\n\nUser: ";
    }
    
    return 0;
}