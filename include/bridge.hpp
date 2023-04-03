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

        struct Params {
            int seed{};
            int n_keep{64};
            int n_ctx{512};
            int n_threads{1};
            int n_batch{16};
            std::size_t last_n_tokens{64};

            constexpr Params& set_seed(int seed) noexcept { this->seed = seed; return *this; }
            constexpr Params& set_number_of_tokens_to_keep(int keep) noexcept { this->n_keep = keep; return *this; }
            constexpr Params& set_number_of_contexts(int ctx) noexcept { this->n_ctx = ctx; return *this; }
            constexpr Params& set_number_of_threads(int threads) noexcept { this->n_threads = threads; return *this; }
            constexpr Params& set_number_of_batches(int batches) noexcept { this->n_batch = batches; return *this; }

            std::optional<FastLlama> build(std::string_view model_id, std::string_view const& filepath);
        };

        // FastLlama(std::string_view model_id, std::string_view const& filepath, int n_threads = 8, int n_ctx = 512, std::size_t last_n_size = 64, int seed = 0, int keep = 64);
        // FastLlama(Model model, int n_threads, std::size_t last_n_size = 64, int seed = 0, int keep = 64);
        FastLlama(FastLlama const&) = default;
        FastLlama(FastLlama &&) noexcept = default;
        FastLlama& operator=(FastLlama const&) = default;
        FastLlama& operator=(FastLlama &&) noexcept = default;
        ~FastLlama() = default;

        bool ingest(std::string prompt);
        bool generate(
            std::function<void(std::string const&)> fn,
            std::size_t num_tokens,
            float top_k,
            float top_p,
            float temp,
            float repeat_penalty,
            std::vector<std::string> const& stop_words = {}
        );

        bool dump_vocab(std::string_view filepath);

        static constexpr Params Builder() noexcept { return {}; }

        constexpr Logger const& get_logger() const noexcept {
            return m_model.logger;
        }

    private:
        auto recycle_embed_if_exceeds_context() -> bool;

        FastLlama() = default;

    private:
        int n_past{};
        int m_seed{};
        int m_keep{};
        int m_batch{64};
        size_t m_mem_per_token{};
        std::string m_model_name;
        std::mt19937 m_rng;
        Model m_model;
        std::vector<token_id_t> m_embd;
        RingBuffer<token_id_t> m_last_n_tokens{64};
        std::vector<float> m_logits;
    };

} // namespace fastllama


#endif // FAST_LLAMA_BRIDGE_HPP