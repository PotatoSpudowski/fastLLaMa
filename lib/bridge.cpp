#include "bridge.hpp"
#include "tokenizer.hpp"
#include "token_buffer.hpp"
#include <unordered_set>
#include <numeric>
#include <chrono>
#include <cmath>
#include "span.hpp"
#include "watermark.hpp"

namespace fastllama {

    void sample_top_k(std::vector<std::pair<double, typename Vocab::id_type>> & logits_id, int top_k) {
        // find the top K tokens
        std::partial_sort(
                logits_id.begin(),
                logits_id.begin() + top_k, logits_id.end(),
                [](auto const & a, auto const & b) { return a.first > b.first; }
        );

        logits_id.resize(static_cast<std::size_t>(top_k));
    }

    auto sample_top_p_top_k(
        Model const& model,
        Span<float> logits,
        RingBuffer<typename Vocab::id_type> const & last_n_tokens,
        double repeat_penalty,
        int top_k,
        double top_p,
        double temp,
        std::mt19937 & rng
    ) -> typename Vocab::id_type {


        std::size_t n_logits = static_cast<std::size_t>(model.params.n_vocab);
        auto const plogits = logits.end() - static_cast<std::ptrdiff_t>(n_logits);

        if (temp <= 0.) {
            auto max_el  = std::max_element(plogits, logits.end());
            return static_cast<typename Vocab::id_type>(std::distance(logits.begin(), max_el));
        }

        std::vector<std::pair<double, typename Vocab::id_type>> logits_id;
        logits_id.resize(n_logits);
        std::unordered_set<typename Vocab::id_type> temp_toks( last_n_tokens.begin(), last_n_tokens.end() );

        {
            const double scale = 1.0 / temp;
            auto const inv_repeat_penalty = 1.0 / repeat_penalty;
            
            auto const temp_n_logits = static_cast<std::ptrdiff_t>(logits_id.size());
            #pragma omp parallel for if (temp_n_logits > 256)
            for (auto i = 0l; i < temp_n_logits; ++i) {
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

        sample_top_k(logits_id, static_cast<int>(top_k > 0 ? std::min(static_cast<std::size_t>(top_k), n_logits) : n_logits));

        double maxl = logits_id[0].first;

        // compute probs for the top K tokens
        std::vector<double> probs;
        probs.resize(logits_id.size());

        double sum{};

        auto const logits_id_size = static_cast<std::ptrdiff_t>(logits_id.size());
        for (auto i = 0l; i < logits_id_size; i++) {
            auto const& kv = logits_id[static_cast<std::size_t>(i)];
            auto p = static_cast<double>(std::exp(static_cast<float>(kv.first - maxl)));
            probs[i] = p;
            sum += p;
        }

        // normalize the probs
        for (auto i = 0ul; i < probs.size(); i++) {
            probs[i] /= sum;
        }

        if (top_p < 1.0) {
            double cumsum{};
            for (auto i = 0ul; i < probs.size(); i++) {
                cumsum += probs[i];
                if (cumsum >= top_p) {
                    probs.resize(i + 1);
                    logits_id.resize(i + 1);
                    break;
                }
            }

        }

        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int idx = dist(rng);

        return logits_id[idx].second;
    }

    std::optional<FastLlama> FastLlama::Params::build(std::string_view const& filepath) {
        auto temp = FastLlama();
        temp.m_model.params.n_ctx = n_ctx;
        temp.m_last_n_tokens = last_n_tokens;
        temp.m_model.set_threads(n_threads);
        temp.m_keep = n_keep;
        temp.m_model.n_batch = n_batch;
        temp.m_model.logger = std::move(logger);
        temp.set_seed(seed);
        temp.m_model.embeddings_eval_enable = embedding_eval_enabled;
        temp.m_model.should_put_all_logits = should_get_all_logits;
        temp.m_model.allocate_extra_mem = allocate_extra_mem;
        temp.m_model.use_mmap = use_mmap;
        temp.m_model.use_mlock = use_mlock;

        printf("\n\n\x1b[32m%s\x1b[0m\n\n", internal::watermark);
        fflush(stdout);

        if (!temp.m_model.load(filepath)) {
            temp.get_logger().log_err("FastLlama::Params::build", "Unable to load model\n");
            return std::nullopt;
        }

        token_id_t inputs[] = { 0, 1, 2, 3 };

        if (!temp.m_model.eval(0, inputs, temp.m_logits, temp.m_mem_per_token)) {
            temp.get_logger().log_err("FastLlama::Params::build", "Unable to evaluate model\n");
            return std::nullopt;
        }

        auto const logits_size = static_cast<std::size_t>(n_ctx * (should_get_all_logits ? temp.m_model.params.n_vocab : 1));
        temp.m_logits.reserve(logits_size);

        if (embedding_eval_enabled) {
            temp.m_model.embeddings.reserve( static_cast<std::size_t>(temp.m_model.params.n_embd) );
        }

        return { std::move(temp) };
    }

    Span<float> FastLlama::get_embeddings() const noexcept {
        if (!m_model.embeddings_eval_enable) get_logger().log_warn(__func__, "Please set the flag `embeddings_eval_enable` to true before getting the embeddings.\n");
        return m_model.embeddings;
    }

    Span<float> FastLlama::get_logits() const noexcept {
        return m_logits;
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

    static auto softmax(std::vector<float> &prob_out, Span<float> logits) {
        if (logits.empty()) return;
        prob_out.resize(logits.size());

        auto const max_val = *std::max_element(logits.begin(), logits.end());
        auto sum_exp = float{};

        for(auto i = std::size_t{}; i < logits.size(); ++i) {
            // Subtract the maximum logit value from the current logit value for numerical stability
            auto const normalized_logit = logits[i] - max_val;
            auto const exp_logit = std::exp(normalized_logit);
            sum_exp += exp_logit;
            prob_out[i] = exp_logit;
        }
        std::transform(prob_out.begin(), prob_out.end(), prob_out.begin(), [sum_exp](float p) { return p / sum_exp; });
    }

    std::optional<float> FastLlama::perplexity(std::string_view prompt) {
        auto old_all_logits = m_model.should_put_all_logits;
        m_model.should_put_all_logits = true;

        auto const tokens = tokenize(m_model.vocabulary, prompt, true);

        auto count = std::size_t{};
        auto const block_size = static_cast<std::size_t>(m_model.n_batch);
        auto const token_len = tokens.size();
        auto const q_blocks = token_len / block_size;
        auto const r_block = token_len % block_size;

        auto const blocks = q_blocks + static_cast<std::size_t>(r_block != 0);
        get_logger().log(__func__, "calculating perplexity over ", blocks, " chunk(s)\n");
        
        double nll{};
        double res{};

        auto block_idx = std::size_t{1};
        std::vector<float> probs;

        for(auto i = std::size_t{}; i < token_len; i += block_size, ++block_idx) {
            auto block = std::min(block_size, token_len - i);

            auto embd_input = Span(tokens.data() + i, block);

            auto const start_time = std::chrono::high_resolution_clock::now();

            if (!m_model.eval(0, embd_input, m_logits, m_mem_per_token)) {
                m_model.should_put_all_logits = old_all_logits;
                return {};
            }
            
            auto const end_time = std::chrono::high_resolution_clock::now();

            if (i == 0) {
                auto const secs = std::chrono::duration<float>(end_time - start_time).count();
                char buff1[20] = {0};
                auto sec_len = snprintf(buff1, sizeof(buff1), "%.2f", secs);
                auto const secs_str = std::string_view{ buff1, static_cast<std::size_t>(sec_len) };
                char buff2[20] = {0};
                auto eta_len = snprintf(buff2, sizeof(buff2), "%.2f", (secs * blocks) / (60.f * 60.f));
                get_logger().log(
                    __func__,
                    secs_str,
                    " seconds per pass - ETA ",
                    std::string_view{ buff2, static_cast<std::size_t>(eta_len) },
                    " hours\n"
                );
            }

            // We get the logits for all the tokens in the context window (params.n_ctx)
            // from llama_eval above.  Now, based on https://huggingface.co/docs/transformers/perplexity,
            // calculate the perplexity over the last half the window (so the model always has
            // some context to predict the token).
            //
            // We rely on the fact that attention in the forward pass only looks at previous
            // tokens here, so the logits returned for each token are an accurate representation
            // of what the model would have predicted at that point.
            //
            // Example, we have a context window of 512, we will compute perplexity for each of the
            // last 256 tokens.  Then, we split the input up into context window size chunks to
            // process the entire prompt.

            auto logits = get_logits();
            auto const vocab_size = static_cast<std::size_t>(m_model.params.n_vocab);
            
            for(auto j = (block >> 1); j < block - 1; ++j) {
                auto const start_pos = j * vocab_size;
                auto const size = std::min(logits.size() - start_pos, vocab_size);
                auto tok_logits = Span( logits.data() + start_pos, size );

                softmax(probs, tok_logits);
                auto const prob = probs[tokens[i + j + 1]];
                nll += static_cast<double>(-std::log(prob));
                ++count;
            }

            res = std::exp(nll / static_cast<double>(count));

            auto const total_end_time = std::chrono::high_resolution_clock::now();
            auto const secs = std::chrono::duration<float>(total_end_time - start_time).count();

            char fstring[64] = {};

            auto len = snprintf(fstring, sizeof(fstring), "[%zu/%zu]: %.4f (took: %.2f secs)\n", block_idx, blocks, res, secs);
            get_logger().log(__func__, std::string_view{ fstring, static_cast<std::size_t>(len) });
        }

        m_model.should_put_all_logits = old_all_logits;
        return { res };
    }

    bool FastLlama::save_state(std::string_view filepath) const noexcept {
        auto writer = BinaryFileWriter(filepath);
        if (!writer) {
            get_logger().log_err(__func__, "unable to open the file saving the model state");
            return false;
        }

        writer.write(&n_past);
        
        std::stringstream ss;
        ss << m_rng;
        auto const rng_str = ss.str();
        auto const rng_str_size = rng_str.size();
        writer.write(&rng_str_size);
        writer.write(rng_str.data(), rng_str_size);

        get_logger().log(__func__, "saving random number generate\n");

        writer.write(&m_mem_per_token);

        std::size_t const embd_size = m_embd.size();
        writer.write(&embd_size);
        writer.write(m_embd.data(), embd_size);

        get_logger().log(__func__, "saving embed vector\n");

        std::size_t m_last_n_tokens_size = m_last_n_tokens.size();
        writer.write(&m_last_n_tokens_size);
        for(auto const& token : m_last_n_tokens) writer.write(&token);

        get_logger().log(__func__, "saving last n tokens\n");

        std::size_t m_logits_size = m_logits.size();
        writer.write(&m_logits_size);
        writer.write(m_logits.data(), m_logits_size);

        get_logger().log(__func__, "saving logits\n");

        std::size_t m_system_prompt_size = m_system_prompt.size();
        writer.write(&m_system_prompt_size);
        writer.write(m_system_prompt.data(), m_system_prompt_size);

        get_logger().log(__func__, "saving system prompt\n");

        return m_model.save_state(writer);
    }

    bool FastLlama::load_state(std::string_view filepath) noexcept {
        auto reader = BinaryFileReader(filepath);
        if (!reader) {
            get_logger().log_err(__func__, "unable to open the file loading the model state");
            return false;
        }

        reader.read(&n_past);

        std::stringstream ss;
        std::size_t rng_str_size;
        reader.read(&rng_str_size);
        std::string rng_str(rng_str_size, '\0');
        reader.read(rng_str.data(), rng_str_size);
        ss << rng_str;
        ss >> m_rng;

        get_logger().log(__func__, "loading random number generator\n");

        reader.read(&m_mem_per_token);

        std::size_t embd_size;
        reader.read(&embd_size);
        m_embd.resize(embd_size);
        reader.read(m_embd.data(), embd_size);

        get_logger().log(__func__, "loading embed vector\n");

        std::size_t m_last_n_tokens_size;
        reader.read(&m_last_n_tokens_size);
        m_last_n_tokens.clear();
        for(auto i = 0ul; i < m_last_n_tokens_size; ++i) {
            token_id_t token;
            reader.read(&token);
            m_last_n_tokens.push_back(token);
        }

        get_logger().log(__func__, "loading last n tokens\n");

        std::size_t m_logits_size;
        reader.read(&m_logits_size);
        m_logits.resize(m_logits_size);
        reader.read(m_logits.data(), m_logits_size);

        get_logger().log(__func__, "loading logits\n");

        std::size_t m_system_prompt_size;
        reader.read(&m_system_prompt_size);
        m_system_prompt.resize(m_system_prompt_size);
        reader.read(m_system_prompt.data(), m_system_prompt_size);

        get_logger().log(__func__, "loading system prompt\n");

        return m_model.load_state(reader);
    }

    bool FastLlama::reset() noexcept {
        get_logger().log(__func__, "resetting the model...\n");
        n_past = 0;
        m_last_n_tokens.clear();
        m_logits.clear();
        m_system_prompt.clear();
        m_embd.clear();
        m_rng = std::mt19937(static_cast<std::size_t>(m_seed));
        auto const res = m_model.reset();
        get_logger().log(__func__, "reset completed.\n");
        return res;
    }

} // namespace fastllama
