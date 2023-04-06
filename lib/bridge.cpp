#include "bridge.hpp"
#include "tokenizer.hpp"
#include "token_buffer.hpp"
#include <unordered_set>
#include <numeric>

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

        logits_id.resize(static_cast<std::size_t>(top_k));
    }

    auto sample_top_p_top_k(
        Model const& model,
        std::vector<float> const& logits,
        RingBuffer<typename Vocab::id> const & last_n_tokens,
        double repeat_penalty,
        int top_k,
        double top_p,
        double temp,
        std::mt19937 & rng
    ) -> typename Vocab::id {


        std::size_t n_logits = static_cast<std::size_t>(model.params.n_vocab);
        auto const plogits = logits.end() - static_cast<std::ptrdiff_t>(n_logits);

        if (temp <= 0.) {
            auto max_el  = std::max_element(plogits, logits.end());
            return static_cast<typename Vocab::id>(std::distance(logits.begin(), max_el));
        }

        std::vector<std::pair<double, typename Vocab::id>> logits_id;
        logits_id.resize(n_logits);
        std::unordered_set<typename Vocab::id> temp_toks( last_n_tokens.begin(), last_n_tokens.end() );

        {
            const double scale = 1.0 / temp;
            auto const inv_repeat_penalty = 1.0 / repeat_penalty;
            
            #pragma omp parallel for if (n_logits > 256)
            for (auto i = 0ul; i < n_logits; ++i) {
                // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
                auto const scaled_logits = static_cast<double>(plogits[static_cast<std::ptrdiff_t>(i)]) * scale;
                if (temp_toks.count(static_cast<int>(i)) != 0) {
                    // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    auto const temp_scaled_logits = scaled_logits * (plogits[static_cast<std::ptrdiff_t>(i)] < 0.0f ? repeat_penalty : inv_repeat_penalty);
                    logits_id[i] = { temp_scaled_logits, i }; 
                } else {
                    logits_id[i] = { scaled_logits, i };
                }
            }
        }

        sample_top_k(logits_id, top_k);

        double maxl = std::max_element(logits_id.begin(), logits_id.end(), [](auto const& p1, auto const& p2) { return p1.first < p2.first; })->first;

        // compute probs for the top K tokens
        std::vector<double> probs;
        probs.reserve(logits_id.size());

        double sum = 0.0;
        auto const logits_id_size = logits_id.size();
        #pragma omp parallel for if(logits_id_size > 256)
        for (const auto & kv : logits_id) {
            auto p = static_cast<double>(expf(static_cast<float>(kv.first - maxl)));
            probs.push_back(p);
            sum += p;
        }

        // normalize the probs
        auto probs_size = probs.size();
        #pragma omp parallel for if(probs_size > 256)
        for (auto i = 0ul; i < probs.size(); ++i) {
            probs[i] /= sum;
        }

        if (top_p < 1.0) {
            double cumsum = 0.0;
            for (auto i = 0ul; i < probs.size(); i++) {
                cumsum += probs[i];
                if (cumsum >= top_p) {
                    probs.resize(i + 1);
                    logits_id.resize(i + 1);
                    break;
                }
            }

            cumsum = 1.0 / cumsum;
            probs_size = probs.size();
            #pragma omp parallel for if(probs_size > 256)
            for (auto i = 0ul; i < probs.size(); i++) {
                probs[i] *= cumsum;
            }
        }

        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int idx = dist(rng);

        return logits_id[static_cast<std::size_t>(idx)].second;
    }

    std::optional<FastLlama> FastLlama::Params::build(std::string_view model_id, std::string_view const& filepath) {
        auto temp = FastLlama();
        temp.m_model.params.n_ctx = n_ctx;
        temp.m_last_n_tokens = last_n_tokens;
        temp.m_model.set_threads(n_threads);
        temp.m_keep = n_keep;
        temp.m_model.n_batch = n_batch;
        temp.m_model.logger = std::move(logger);
        temp.set_seed(seed);

        if (!temp.m_model.load(model_id, filepath, is_old_model)) {
            temp.get_logger().log_err("FastLlama::Params::build", "Unable to load model\n");
            return std::nullopt;
        }

        if (!temp.m_model.eval(0, { 0, 1, 2, 3 }, temp.m_logits, temp.m_mem_per_token)) {
            temp.get_logger().log_err("FastLlama::Params::build", "Unable to evaluate model\n");
            return std::nullopt;
        }
        return { std::move(temp) };
    }

    auto FastLlama::recycle_embed_if_exceeds_context() -> bool {
        auto const len = static_cast<int>(m_embd.size());
        if (len <= 0) return false;

        if (len + n_past <= m_model.params.n_ctx) return false;

        auto const remaining = n_past - m_keep;
        if (remaining <= 0) return false;

        n_past = m_keep;
        auto const system_prompt_size = static_cast<std::ptrdiff_t>(m_system_prompt.size());
        auto last_tokens_begin = m_last_n_tokens.begin() + m_model.params.n_ctx + (remaining >> 1) - len;
        m_embd.insert(m_embd.begin(), last_tokens_begin, m_last_n_tokens.end() - len - system_prompt_size);
        m_embd.insert(m_embd.begin(), m_system_prompt.begin(), m_system_prompt.end());
        return true;
    }

    bool FastLlama::dump_vocab(std::string_view filepath) {
        return m_model.dump_vocab(filepath);
    }

    bool FastLlama::ingest(std::string prompt, bool is_system_prompt) {
        m_model.logger.reset();
        if (!m_model.is_valid) {
            m_model.logger.log_err("FastLlama::ingest", "tried to ingest using invalid model");
            return false;
        }

        prompt.insert(0, 1, ' ');

        auto embd_input = tokenize(m_model.vocabulary, prompt, true);
        
        auto max_input_size = m_model.params.n_ctx - 4;
        if (embd_input.size() > static_cast<std::size_t>(max_input_size)) {
            m_model.logger.log_err("ingest", "prompt size(='", embd_input.size(), "') exceeds maximum allowed size('", max_input_size, "')");
            return false;
        }
        
        if (is_system_prompt) {
            if (m_keep < static_cast<int>(embd_input.size())) {
                m_model.logger.log_err("ingest", "system prompt size(='", embd_input.size(), "') exceeds 'n_keep'(='", m_keep, "')");
                return false;
            }
            m_system_prompt = embd_input;
        }

        auto const n_batch = m_model.n_batch;

        for(auto i = 0ul; i < embd_input.size(); i += static_cast<std::size_t>(n_batch)) {
            auto block = std::min(static_cast<std::size_t>(n_batch), embd_input.size() - i);

            recycle_embed_if_exceeds_context();

            // std::cout<<"E Size: " << m_embd.size()<<", Past: "<<n_past<<", Mem: "<<m_mem_per_token<<std::endl;

            if (!m_embd.empty()) {
                if (!m_model.eval(static_cast<std::size_t>(n_past), m_embd, m_logits, m_mem_per_token)) {
                    return false;
                }
            }

            n_past += m_embd.size();
            m_embd.clear();

            std::copy_n(embd_input.begin() + static_cast<std::ptrdiff_t>(i), block, std::back_inserter(m_embd));
            std::copy_n(embd_input.begin() + static_cast<std::ptrdiff_t>(i), block, std::back_inserter(m_last_n_tokens));
        }

        m_last_n_tokens.clear();
        return true;
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
        m_model.logger.reset();
        if (!m_model.is_valid) {
            m_model.logger.log_err("FastLlama::generate", "tried to generate using invalid model");
            return false;
        }
        auto const max_token_buffer_size = std::accumulate(
            stop_words.begin(),
            stop_words.end(),
            std::size_t{},
            [&vocab=m_model.vocabulary](auto const& max_el, auto const& el) {
                return std::max(max_el, tokenize(vocab, el, false).size());
            }
        );

        auto token_buffer = TokenBuffer(m_model.vocabulary, max_token_buffer_size, [&fn](auto&& s) {
            fn(std::forward<decltype(s)>(s));
        });

        // auto new_line_token = tokenize(m_model.vocabulary, "\n", false);
        // auto new_line_token_id = new_line_token.front();

        for (auto i = 0ul; i < num_tokens; ++i) {
            auto const [is_stop_token_present, to_be_flush_substr] = token_buffer.are_tokens_present_in_buffer(stop_words);
            
            if (is_stop_token_present) {
                fn(std::string(to_be_flush_substr));
                return true;
            }

            recycle_embed_if_exceeds_context();

            if (!m_embd.empty()) {
                if (!m_model.eval(static_cast<std::size_t>(n_past), m_embd, m_logits, m_mem_per_token)) {
                    return false;
                }
            }

            n_past += m_embd.size();
            m_embd.clear();

            auto token_id = sample_top_p_top_k(
                m_model,
                m_logits,
                m_last_n_tokens,
                static_cast<double>(repeat_penalty),
                static_cast<int   >(top_k),
                static_cast<double>(top_p),
                static_cast<double>(temp),
                m_rng
            );
            if (token_id == FastLlama::EOS) break;
            m_last_n_tokens.push_back(token_id);
            token_buffer.add(token_id);
            m_embd.push_back(token_id);
        }

        token_buffer.flush_buffer();

        return true;
    }

} // namespace fastllama
