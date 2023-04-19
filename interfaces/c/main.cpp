#include "fastllama.h"
#include "bridge.hpp"
#include <stdarg.h>
#include <stdio.h>

struct llama_model_context {
    std::optional<fastllama::FastLlama> inner{std::nullopt};
    std::vector<std::string> stop_words{};
    fastllama::FastLlama::Params builder{};
};

inline static LLAMA_LOGGER_FUNC make_def_info_logger_func() {
    return +[](char const* func_name, int func_name_size, char const* message, int message_size) {
        printf("\x1b[32;1m[Info]:\x1b[0m \x1b[32mFunc('%.*s') %.*s\x1b[0m", func_name_size, func_name, message_size, message);
        fflush(stdout);
    };
}

inline static LLAMA_LOGGER_FUNC make_def_err_logger_func() {
    return +[](char const* func_name, int func_name_size, char const* message, int message_size) {
        fprintf(stderr, "\x1b[31;1m[Error]:\x1b[0m \x1b[31mFunc('%.*s') %.*s\x1b[0m", func_name_size, func_name, message_size, message);
        fflush(stdout);
    };
}

inline static LLAMA_LOGGER_FUNC make_def_warn_logger_func() {
    return +[](char const* func_name, int func_name_size, char const* message, int message_size) {
        printf("\x1b[93;1m[Warn]:\x1b[0m \x1b[93mFunc('%.*s') %.*s\x1b[0m", func_name_size, func_name, message_size, message);
        fflush(stdout);
    };
}

inline static LLAMA_LOGGER_RESET_FUNC make_def_reset_logger_func() {
    return +[]() {
        printf("\x1b[0m");
        fflush(stdout);
    };
}

#ifdef __cplusplus
extern "C" {
#endif

    struct llama_model_context_args llama_create_default_context_args() {
        struct llama_model_context_args result{};
        auto builder = fastllama::FastLlama::builder();
        result.last_n_tokens = builder.last_n_tokens;
        result.logger.log = make_def_info_logger_func();
        result.logger.log_err = make_def_err_logger_func();
        result.logger.log_warn = make_def_warn_logger_func();
        result.logger.reset = make_def_reset_logger_func();

        result.n_batch = builder.n_batch;
        result.n_ctx = builder.n_ctx;
        result.n_keep = builder.n_keep;
        result.n_threads = builder.n_threads;
        result.seed = builder.seed;
        result.allocate_extra_mem = 0ul;
        result.embedding_eval_enabled = false;
        result.should_get_all_logits = false;
        return result;
    }

    struct llama_model_context* llama_create_context(struct llama_model_context_args arg) {
        using namespace fastllama;
        auto builder = FastLlama::builder();
        builder.last_n_tokens = arg.last_n_tokens;
        builder.n_batch = arg.n_batch;
        builder.n_ctx = arg.n_ctx;
        builder.n_keep = arg.n_keep;
        builder.n_threads = arg.n_threads;
        builder.seed = arg.seed;
        builder.allocate_extra_mem = arg.allocate_extra_mem;
        builder.should_get_all_logits = arg.should_get_all_logits;
        builder.embedding_eval_enabled = arg.embedding_eval_enabled;

        auto def_logger = DefaultLogger{};
        def_logger.log = arg.logger.log;
        def_logger.log_err = arg.logger.log_err;
        def_logger.log_warn = arg.logger.log_warn;
        def_logger.reset = arg.logger.reset;
        
        builder.logger = Logger(std::move(def_logger));

        auto result = new llama_model_context();
        if (result) {
            result->builder = std::move(builder);
            result->inner = std::nullopt;
        }
        return result;
    }

    inline static constexpr bool is_model_valid(struct llama_model_context const* const ctx) noexcept {
        if (ctx == nullptr) {
            fprintf(stderr, "model context is not initalized. Please use `llama_create_context` to create a context.\n");
            return false;
        }

        if (ctx->inner == std::nullopt) {
            fprintf(stderr, "model is not loaded. Please use `llama_load_model` to load a model.\n");
            return false;
        }
        
        return true;
    }

    bool llama_load_model_str(struct llama_model_context* model_context, char const* model_id, char const* filepath) {
        if (model_context == nullptr) {
            fprintf(stderr, "model context is not initalized. Please use `llama_create_context` to create a context.\n");
            return false;
        }

        if (model_context->inner != std::nullopt) {
            fprintf(stderr, "model is already loaded.\n");
            return false;
        }

        auto maybe_model = model_context->builder.build(model_id, filepath);
        if (!maybe_model) return false;
        model_context->inner = std::move(maybe_model);

        return true;
    }

    bool llama_load_model(struct llama_model_context* model_context, enum ModelKind model_id, char const* filepath) {
        if (model_context == nullptr) {
            fprintf(stderr, "model context is not initalized. Please use `llama_create_context` to create a context.\n");
            return false;
        }

        if (model_context->inner != std::nullopt) {
            fprintf(stderr, "model is already loaded.\n");
            return false;
        }

        fastllama::ModelKind id{};

        switch (model_id) {
            case LLAMA_7B: id = fastllama::ModelKind::LLAMA_7B; break;
            case LLAMA_13B: id = fastllama::ModelKind::LLAMA_13B; break;
            case LLAMA_30B: id = fastllama::ModelKind::LLAMA_30B; break;
            case LLAMA_65B: id = fastllama::ModelKind::LLAMA_65B; break;
            case ALPACA_LORA_7B: id = fastllama::ModelKind::ALPACA_LORA_7B; break;
            case ALPACA_LORA_13B: id = fastllama::ModelKind::ALPACA_LORA_13B; break;
            case ALPACA_LORA_30B: id = fastllama::ModelKind::ALPACA_LORA_30B; break;
            case ALPACA_LORA_65B: id = fastllama::ModelKind::ALPACA_LORA_65B; break;
            default: {
                fprintf(stderr, "invalid model id.\n");
                return false;
            }
        }

        auto maybe_model = model_context->builder.build(id, filepath);
        if (!maybe_model) return false;
        model_context->inner = std::move(maybe_model);

        return true;
    }

    bool llama_set_stop_words(struct llama_model_context* model_context, char const** words, size_t len) {
        if (model_context == nullptr) {
            fprintf(stderr, "model context is not initalized. Please use `llama_create_context` to create a context.\n");
            return false;
        }

        model_context->stop_words.resize(len);

        for (auto i = std::size_t{}; i < len; i++) {
            model_context->stop_words[i] = words[i];
        }

        return true;
    }

    bool llama_ingest(struct llama_model_context* model_context, char const* prompt) {
        if (!is_model_valid(model_context)) return false;

        return model_context->inner->ingest(std::string(prompt), false);
    }

    bool llama_ingest_system_prompt(struct llama_model_context* model_context, char const* prompt) {
        if (!is_model_valid(model_context)) return false;

        return model_context->inner->ingest(std::string(prompt), true);
    }

    bool llama_generate(
        struct llama_model_context* model_context,
        LLAMA_STREAM_FUNC stream_fn,
        size_t number_of_tokens,
        float top_k,
        float top_p,
        float temp,
        float repeat_penalty
    ) {
        if (!is_model_valid(model_context)) return false;

        return model_context->inner->generate([stream_fn](std::string const& s) {
            stream_fn(s.data(), static_cast<int>(s.size()));
        }, number_of_tokens, top_k, top_p, temp, repeat_penalty, model_context->stop_words);
    }

    void llama_free_context(struct llama_model_context* ctx) {
        delete ctx;
    }

    float llama_perplexity(struct llama_model_context* model_context, char const* prompt) {
        if (!is_model_valid(model_context)) return -1;

        auto temp_res = model_context->inner->perplexity(prompt);

        return temp_res.value_or(-1);
    }

    llama_array_view_f llama_get_embeddings(struct llama_model_context const* const model_context) {
        if (!is_model_valid(model_context)) return { nullptr, 0ul };
        auto const& arr = model_context->inner->get_embeddings();
        return llama_array_view_f{ arr.data(), arr.size() };
    }

    llama_array_view_f llama_get_logits(struct llama_model_context const* const model_context) {
        if (!is_model_valid(model_context)) return { nullptr, 0ul };
        auto const& arr = model_context->inner->get_logits();
        return llama_array_view_f{ arr.data(), arr.size() };
    }

    bool llama_save_state(struct llama_model_context* model_context, char const* filepath) {
        if (!is_model_valid(model_context)) return false;
        return model_context->inner->save_state(filepath);
    }
    
    bool llama_load_state(struct llama_model_context* model_context, char const* filepath) {
        if (!is_model_valid(model_context)) return false;
        return model_context->inner->load_state(filepath);
    }

    bool llama_attach_lora(struct llama_model_context* model_context, char const* filepath) {
        if (!is_model_valid(model_context)) return false;
        return model_context->inner->attach_lora(filepath);
    }

    bool llama_detach_lora(struct llama_model_context* model_context) {
        if (!is_model_valid(model_context)) return false;
        return model_context->inner->detach_lora();
    }

    bool llama_reset_model(struct llama_model_context* model_context) {
        if (!is_model_valid(model_context)) return false;
        return model_context->inner->reset();
    }

    void llama_handle_signal(int) {
        printf("Quitting the app...");
        exit(0);
    }

#ifdef __cplusplus
}
#endif
