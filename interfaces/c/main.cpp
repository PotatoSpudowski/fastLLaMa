#include "fastllama.h"
#include "bridge.hpp"
#include <stdarg.h>

struct llama_model_context {
    std::optional<fastllama::FastLlama> inner;
    std::vector<std::string> stop_words;
    fastllama::FastLlama::Params builder;
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
        return result;
    }

    struct llama_model_context* llama_create_context(struct llama_model_context_args arg) {
        auto builder = fastllama::FastLlama::builder();
        builder.last_n_tokens = arg.last_n_tokens;
        builder.n_batch = arg.n_batch;
        builder.n_ctx = arg.n_ctx;
        builder.n_keep = arg.n_keep;
        builder.n_threads = arg.n_threads;
        builder.seed = arg.seed;

        auto def_logger = fastllama::DefaultLogger{};
        def_logger.log = arg.logger.log;
        def_logger.log_err = arg.logger.log_err;
        def_logger.log_warn = arg.logger.log_warn;
        def_logger.reset = arg.logger.reset;

        builder.logger = fastllama::Logger(std::move(def_logger));

        auto result = new llama_model_context();
        result->builder = builder;
        result->inner = std::nullopt;
        return result;
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

    bool llama_set_stop_words(struct llama_model_context* model_context, int number_of_words, ...) {
        if (model_context == nullptr) {
            fprintf(stderr, "model context is not initalized. Please use `llama_create_context` to create a context.\n");
            return false;
        }

        va_list args;
        va_start(args, number_of_words);
        model_context->stop_words.resize(number_of_words);

        for (auto i = 0; i < number_of_words; i++) {
            std::string arg = va_arg(args, char const*);
            model_context->stop_words[static_cast<std::size_t>(i)] = std::move(arg);
        }

        va_end(args);
        return true;
    }

    bool llama_ingest(struct llama_model_context* model_context, char const* prompt) {
        if (model_context == nullptr) {
            fprintf(stderr, "model context is not initalized. Please use `llama_create_context` to create a context.\n");
            return false;
        }

        if (model_context->inner == std::nullopt) {
            fprintf(stderr, "model is not loaded. Please use `llama_load_model` to load a model.\n");
            return false;
        }

        return model_context->inner->ingest(std::string(prompt), false);
    }

    bool llama_ingest_system_prompt(struct llama_model_context* model_context, char const* prompt) {
        if (model_context == nullptr) {
            fprintf(stderr, "model context is not initalized. Please use `llama_create_context` to create a context.\n");
            return false;
        }

        if (model_context->inner == std::nullopt) {
            fprintf(stderr, "model is not loaded. Please use `llama_load_model` to load a model.\n");
            return false;
        }

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
        if (model_context == nullptr) {
            fprintf(stderr, "model context is not initalized. Please use `llama_create_context` to create a context.\n");
            return false;
        }

        if (model_context->inner == std::nullopt) {
            fprintf(stderr, "model is not loaded. Please use `llama_load_model` to load a model.\n");
            return false;
        }

        return model_context->inner->generate([stream_fn](std::string const& s) {
            stream_fn(s.data(), static_cast<int>(s.size()));
        }, number_of_tokens, top_k, top_p, temp, repeat_penalty, model_context->stop_words);
    }

    void llama_free_context(struct llama_model_context* ctx) {
        delete ctx;
    }

#ifdef __cplusplus
}
#endif
