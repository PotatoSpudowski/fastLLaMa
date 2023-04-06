#if !defined(FAST_LLAMA_BRIDGE_H)
#define FAST_LLAMA_BRIDGE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void(*LLAMA_LOGGER_FUNC)(char const* function_name, int function_name_size, char const* message, int message_size);
typedef void(*LLAMA_LOGGER_RESET_FUNC)();
typedef void(*LLAMA_STREAM_FUNC)(char const* token_stream, int token_stream_size);

struct llama_model_context;

enum ModelKind {
    LLAMA_7B = 0,
    LLAMA_13B,
    LLAMA_30B,
    LLAMA_65B,
    ALPACA_LORA_7B,
    ALPACA_LORA_13B,
    ALPACA_LORA_30B,
    ALPACA_LORA_65B,
};

struct llama_logger {
    LLAMA_LOGGER_FUNC log; // info log
    LLAMA_LOGGER_FUNC log_err; // error log
    LLAMA_LOGGER_FUNC log_warn; // error log
    LLAMA_LOGGER_RESET_FUNC reset; // reset the log or anything else
};

struct llama_model_context_args {
    int seed;
    int n_keep;
    int n_ctx;
    int n_threads;
    int n_batch;
    size_t last_n_tokens;
    struct llama_logger logger;
};

struct llama_model_context_args llama_create_default_context_args();
struct llama_model_context* llama_create_context(struct llama_model_context_args arg);

bool llama_load_model(struct llama_model_context* model_context, enum ModelKind kind, char const* filepath);
bool llama_load_model_str(struct llama_model_context* model_context, char const* model_id, char const* filepath);

bool llama_set_stop_words(struct llama_model_context* model_context, int number_of_words, ...);

bool llama_ingest_system_prompt(struct llama_model_context* model_context, char const* prompt);
bool llama_ingest(struct llama_model_context* model_context, char const* prompt);

bool llama_generate(
    struct llama_model_context* model_context,
    LLAMA_STREAM_FUNC stream_fn,
    size_t number_of_tokens,
    float top_k,
    float top_p,
    float temp,
    float repeat_penalty
);

void llama_free_context(struct llama_model_context*);

#ifdef __cplusplus
}
#endif

#endif // FAST_LLAMA_BRIDGE_H
