#include "bridge.hpp"
#include "tokenizer.hpp"

namespace fastllama {

    void sample_top_k(std::vector<std::pair<double, typename Vocab::id>> & logits_id, int top_k) {
        // find the top K tokens
        std::partial_sort(
                logits_id.begin(),
                logits_id.begin() + top_k, logits_id.end(),
                [](const std::pair<double, typename Vocab::id> & a, const std::pair<double, typename Vocab::id> & b) {
                    return a.first > b.first;
                }
        );

        logits_id.resize(top_k);
    }

    auto llama_sample_top_p_top_k(
        const fastllama::Vocab & vocab,
        const float * logits,
        RingBuffer<typename Vocab::id> const & last_n_tokens,
        double repeat_penalty,
        int top_k,
        double top_p,
        double temp,
        std::mt19937 & rng
    ) -> typename Vocab::id {

        int n_logits = vocab.id_to_token.size();

        std::vector<std::pair<double, typename fastllama::Vocab::id>> logits_id;
        logits_id.reserve(n_logits);

        {
            const double scale = 1.0 / temp;
            auto const inv_repeat_penalty = 1.0 / repeat_penalty;
            
            #pragma omp parallel for
            for (int i = 0; i < n_logits; ++i) {
                // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
                auto const scaled_logits = logits[i] * scale;
                if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
                    // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    auto const is_less_than_zero = logits[i] < 0.0;
                    auto const temp = scaled_logits * (is_less_than_zero * repeat_penalty + (!is_less_than_zero * inv_repeat_penalty));
                    logits_id.push_back(std::make_pair(temp, i));             
                } else {
                    logits_id.push_back(std::make_pair(scaled_logits, i));
                }
            }
        }

        sample_top_k(logits_id, top_k);

        double maxl = std::max_element(logits_id.begin(), logits_id.end(), [](auto const& p1, auto const& p2) { return p1.first < p2.first; })->first;

        // compute probs for the top K tokens
        std::vector<double> probs;
        probs.reserve(logits_id.size());

        double sum = 0.0;
        for (const auto & kv : logits_id) {
            double p = exp(kv.first - maxl);
            probs.push_back(p);
            sum += p;
        }

        // normalize the probs

        #pragma omp parallel for
        for (auto i = 0ul; i < probs.size(); ++i) {
            probs[i] /= sum;
        }

        if (top_p < 1.0f) {
            double cumsum = 0.0f;
            for (auto i = 0ul; i < probs.size(); i++) {
                cumsum += probs[i];
                if (cumsum >= top_p) {
                    probs.resize(i + 1);
                    logits_id.resize(i + 1);
                    break;
                }
            }

            cumsum = 1.0/cumsum;
            #pragma omp parallel for
            for (int i = 0ul; i < probs.size(); i++) {
                probs[i] *= cumsum;
            }
        }

        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int idx = dist(rng);

        return logits_id[idx].second;
    }

    
    FastLlama::FastLlama(std::string_view model_id, std::string_view const& filepath, int n_threads, int n_ctx, std::size_t last_n_size, int seed)
        : m_threads(n_threads)
        , m_seed(seed)
        , m_last_n_tokens(last_n_size)
    {
        m_model.params.n_ctx = n_ctx;
        if (!m_model.load(model_id, filepath)) {
            throw std::runtime_error("Unable to load model");
        }

        if (!m_model.eval(m_threads, 0, { 0, 1, 2, 3 }, m_logits, m_mem_per_token)) {
            throw std::runtime_error("Unable to evaluate model");
        }
    }

    FastLlama::FastLlama(Model model, int n_threads, std::size_t last_n_size, int seed)
        : m_threads(n_threads)
        , m_seed(seed)
        , m_model(std::move(model))
        , m_last_n_tokens(last_n_size)
    {
        if (!m_model.eval(m_threads, 0, { 0, 1, 2, 3 }, m_logits, m_mem_per_token)) {
            throw std::runtime_error("Unable to evaluate model");
        }
    }

    bool FastLlama::ingest(std::string prompt) {
        prompt.insert(0, 1, ' ');

        auto embd_input = tokenize(m_model.vocabulary, prompt, true);

    }

    bool FastLlama::generate(
        std::function<void(std::string const&)> fn,
        std::size_t num_tokens,
        float top_k,
        float top_p,
        float temp,
        float repeat_penalty,
        std::vector<std::string> const& stop_words
    ) {

    }

} // namespace fastllama
