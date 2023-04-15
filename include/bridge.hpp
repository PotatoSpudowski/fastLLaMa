#if !defined(FAST_LLAMA_BRIDGE_HPP)
#define FAST_LLAMA_BRIDGE_HPP

#include <string>
#include <vector>
#include <random>
#include "llama.hpp"
#include "vocab.hpp"
#include "logger.hpp"
#include "ring_buffer.hpp"
#include <optional>

namespace fastllama {    
    struct FastLlama {
        using token_id_t = typename Vocab::id;

        static constexpr token_id_t EOS = 2;
        static constexpr token_id_t BOS = 1;

        struct Params {
            int seed{};
            int n_keep{64};
            int n_ctx{512};
            int n_threads{1};
            int n_batch{16};
            bool is_old_model{false};
            bool embedding_eval_enabled{false};
            bool should_get_all_logits{false};
            std::size_t last_n_tokens{64};
            std::size_t allocate_extra_mem{};
            Logger logger{};

            constexpr Params& set_seed(int in_seed) noexcept { this->seed = in_seed; return *this; }
            constexpr Params& set_number_of_tokens_to_keep(int keep) noexcept { this->n_keep = keep; return *this; }
            constexpr Params& set_number_of_contexts(int ctx) noexcept { this->n_ctx = ctx; return *this; }
            constexpr Params& set_number_of_threads(int threads) noexcept { this->n_threads = threads; return *this; }
            constexpr Params& set_number_of_batches(int batches) noexcept { this->n_batch = batches; return *this; }
            constexpr Params& set_is_old_model(bool flag) noexcept { this->is_old_model = flag; return *this; }
            constexpr Params& set_embedding_eval_enabled(bool flag) noexcept { this->embedding_eval_enabled = flag; return *this; }
            constexpr Params& set_should_get_all_logits(bool flag) noexcept { this->should_get_all_logits = flag; return *this; }
            constexpr Params& set_allocate_extra_mem(std::size_t allocate_extra_mem) noexcept { this->allocate_extra_mem = allocate_extra_mem; return *this; }
            Params& set_logger(Logger in_logger) noexcept { this->logger = std::move(in_logger); return *this; }

            std::optional<FastLlama> build(std::string_view model_id, std::string_view const& filepath);
            std::optional<FastLlama> build(ModelKind model_id, std::string_view const& filepath);
        };

        // FastLlama(std::string_view model_id, std::string_view const& filepath, int n_threads = 8, int n_ctx = 512, std::size_t last_n_size = 64, int seed = 0, int keep = 64);
        // FastLlama(Model model, int n_threads, std::size_t last_n_size = 64, int seed = 0, int keep = 64);
        FastLlama(FastLlama const&) = delete;
        FastLlama(FastLlama &&) noexcept = default;
        FastLlama& operator=(FastLlama const&) = delete;
        FastLlama& operator=(FastLlama &&) noexcept = default;
        ~FastLlama() { m_model.unload(); }

        bool ingest(std::string prompt, bool is_system_prompt = false);
        bool generate(
            std::function<void(std::string const&)> fn,
            std::size_t num_tokens,
            float top_k,
            float top_p,
            float temp,
            float repeat_penalty,
            std::vector<std::string> const& stop_words = {}
        );

        std::optional<float> perplexity(std::string_view prompt);

        Span<float> get_embeddings() const noexcept;
        Span<float> get_logits() const noexcept;

        bool dump_vocab(std::string_view filepath);

        static Params builder() noexcept { return {}; }

        constexpr Logger const& get_logger() const noexcept {
            return m_model.logger;
        }

    private:
        auto recycle_embed_if_exceeds_context() -> bool;

        FastLlama() = default;

        void set_seed(int seed) { m_rng = std::mt19937(static_cast<std::uint32_t>(seed)); m_seed = seed; }

    private:
        int n_past{};
        int m_seed{};
        int m_keep{};
        size_t m_mem_per_token{};
        std::string m_model_name;
        std::mt19937 m_rng;
        Model m_model;
        std::vector<token_id_t> m_embd;
        RingBuffer<token_id_t> m_last_n_tokens{64};
        std::vector<float> m_logits;
        std::vector<token_id_t> m_system_prompt;
    };

} // namespace fastllama


#endif // FAST_LLAMA_BRIDGE_HPP
