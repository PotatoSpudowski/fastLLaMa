#if !defined(FAST_LLAMA_BRIDGE_H)
#define FAST_LLAMA_BRIDGE_H

#include <stddef.h>

typedef void(*LLAMA_LOGGER_FUNC)(char const* function_name, int function_name_size, char const* message, int message_size);

struct llama_model_context;

struct llama_str;

struct llama_logger {
    LLAMA_LOGGER_FUNC log; // info log
    LLAMA_LOGGER_FUNC log_err; // error log
    LLAMA_LOGGER_FUNC log_warn; // error log
};

void llama_set_logger(llama_logger const*);

struct llama_model_context* llama_create_context(int n_threads, size_t n_ctx, size_t last_n_size, int seed);

void llama_load_model(struct llama_model_context* model_context, char const* model_id, char const* filepath);

void llama_set_stop_words(struct llama_model_context* model_context, ...);

void llama_ingest_prompt(struct llama_model_context* model_context, char const* prompt);

void llama_generate(
    struct llama_model_context* model_context,
    size_t number_of_tokens,
    float top_k,
    float top_p,
    float temp,
    float repeat_penalty
);

#endif // FAST_LLAMA_BRIDGE_H
