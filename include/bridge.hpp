#if !defined(FAST_LLAMA_BRIDGE_HPP)
#define FAST_LLAMA_BRIDGE_HPP

#include <string>
#include <vector>
#include <random>
#include "llama.hpp"
#include "vocab.hpp"
#include "ring_buffer.hpp"

namespace fastllama {
    
    struct FastLlama {
        using token_id_t = typename Vocab::id;

        FastLlama(std::string_view model_id, std::string_view const& filepath, int n_threads, int n_ctx = 512, std::size_t last_n_size = 64, int seed = 0);
        FastLlama(Model model, int n_threads, std::size_t last_n_size = 64, int seed = 0);
        bool ingest(std::string prompt);
        bool generate(
            std::function<void(std::string const&)> fn,
            std::size_t num_tokens,
            float top_k,
            float top_p,
            float temp,
            float repeat_penalty,
            std::vector<std::string> const& stop_words
        );

        // bool generate(
        //     std::function<void(char const*)> fn,
        //     std::size_t num_tokens,
        //     float top_k,
        //     float top_p,
        //     float temp,
        //     float repeat_penalty,
        //     char const** stop_words,
        //     std::size_t n_stop_words
        // );

    private:
        int m_threads;
        int n_past{0};
        int m_seed{0};
        size_t m_mem_per_token = 0;
        std::string m_model_name;
        std::mt19937 m_rng;
        Model m_model;
        fastllama::Vocab m_vocab;
        std::vector<token_id_t> m_embd;
        RingBuffer<token_id_t> m_last_n_tokens;
        std::vector<float> m_logits;
    };

} // namespace fastllama


#endif // FAST_LLAMA_BRIDGE_HPP
