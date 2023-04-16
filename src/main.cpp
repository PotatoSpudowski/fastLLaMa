#include "bridge.hpp"
#include <thread>
#include <chrono>

using namespace std::chrono_literals;
using namespace fastllama;

int main() {
    auto maybe_bridge = fastllama::FastLlama::builder()
        .set_number_of_threads(16)
        .set_number_of_batches(64)
        .set_number_of_contexts(512)
        .set_number_of_tokens_to_keep(48)
        .set_should_get_all_logits(true)
        .build(fastllama::ModelKind::ALPACA_LORA_7B, "./models/ALPACA-LORA-7B/alpaca-lora-q4_0.bin");
    
    if (!maybe_bridge) {
        return 1;
    }
    auto bridge = std::move(maybe_bridge.value());
    // bridge.dump_vocab("./vocab.txt");
    auto& logger = bridge.get_logger();

    // logger.log_warn("main", "Ingesting, please wait...\n");
    // bridge.ingest(
    //     R"(Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.
    //     User: Hello, Bob.
    //     Bob: Hello. How may I help you today?
    //     User: Please tell me the largest city in Europe.
    //     Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.)"
    // );
    // logger.log_warn("main", "Ingestion complete!\n");

    // bridge.save_state("./models/fast_llama.bin");
    // bridge.load_state("./models/fast_llama.bin");
    std::string prompt;

    std::cout<<"User: ";

    while(std::getline(std::cin, prompt)) {
        if (prompt == "save") {
            bridge.save_state("./models/fast_llama.bin");
            std::cout<<"User: ";
            continue;
        } else if (prompt == "exit") {
            exit(0);
        } else if (prompt == "load") {
            bridge.load_state("./models/fast_llama.bin");
            std::cout<<"User: ";
            continue;
        }
        prompt = "\n\n### Instruction:\n\n" + prompt + "\n\n### Response:\n\n";
        
        bridge.ingest(prompt);

        bridge.generate([](std::string const& s) {
            std::cout<<s;
            std::cout.flush();
        }, 300, 40, 0.95f, 0.8f, 1.0f, { "###" });
        
        std::cout<<"\nUser: ";
    }
    
    return 0;
}