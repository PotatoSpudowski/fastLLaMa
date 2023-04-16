#include "bridge.hpp"
#include <thread>
#include <chrono>
#include "concurrency/pool.hpp"

using namespace std::chrono_literals;
using namespace fastllama;

std::mutex mu;
struct LoadWorker {
    void operator()() noexcept {
        std::lock_guard lk{mu};
        std::cout<<"Hello Worker: "<<std::this_thread::get_id()<<"\n";
    }
};

int main() {
    // ThreadPool<LoadWorker> w(2);
    // for(auto i = 0ul; i < 10; ++i) {
    //     w.add_work({});
    // }

    // w.wait();
    auto maybe_bridge = fastllama::FastLlama::builder()
        .set_number_of_threads(16)
        .set_number_of_batches(64)
        .set_number_of_contexts(512)
        .set_number_of_tokens_to_keep(48)
        .build(fastllama::ModelKind::LLAMA_7B, "./models/7B/ggml-model-q4_0.bin");
    
    if (!maybe_bridge) {
        return 1;
    }
    auto bridge = std::move(maybe_bridge.value());
    // bridge.dump_vocab("./vocab.txt");
    auto& logger = bridge.get_logger();

    logger.log_warn("main", "Ingesting, please wait...\n");
    bridge.ingest(
        R"(Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.
        User: Hello, Bob.
        Bob: Hello. How may I help you today?
        User: Please tell me the largest city in Europe.
        Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.)"
    );
    logger.log_warn("main", "Ingestion complete!\n");

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
        prompt = "User: " + prompt;
        
        bridge.ingest(prompt);

        bridge.generate([](std::string const& s) {
            std::cout<<s;
            std::cout.flush();
        }, 50, 40, 0.95f, 0.8f, 1.0f, { "User: " });
        
        std::cout<<"\nUser: ";
    }
    
    return 0;
}