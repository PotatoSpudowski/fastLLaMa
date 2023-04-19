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
    LLAMA_7B = 0,   // "LLAMA-7B"
    LLAMA_13B,      // "LLAMA-13B"
    LLAMA_30B,      // "LLAMA-30B"
    LLAMA_65B,      // "LLAMA-65B"
    ALPACA_LORA_7B, // "ALPACA-LORA-7B"
    ALPACA_LORA_13B,// "ALPACA-LORA-13B"
    ALPACA_LORA_30B,// "ALPACA-LORA-30B"
    ALPACA_LORA_65B,// "ALPACA-LORA-65B"
};

struct llama_logger {
    LLAMA_LOGGER_FUNC log; // info log
    LLAMA_LOGGER_FUNC log_err; // error log
    LLAMA_LOGGER_FUNC log_warn; // error log
    LLAMA_LOGGER_RESET_FUNC reset; // reset the log or anything else
};

// Provides a view of an array of floats.
struct llama_array_view_f {
    float const* const data;
    size_t size;
};

// Arguments to the model for creating a context. This helps us from having very large function parament
// and allows us to have default arguments.
struct llama_model_context_args {
    bool embedding_eval_enabled;
    bool should_get_all_logits;
    int seed;
    int n_keep;
    int n_ctx;
    int n_threads;
    int n_batch;
    size_t last_n_tokens;
    size_t allocate_extra_mem;
    struct llama_logger logger;
};

// Creates the default context arguments to reduce the initialization complexity.
struct llama_model_context_args llama_create_default_context_args();

/**
 * @brief Creates the model context that is used for storing the state.
 * 
 * @param args is of type `llama_model_context_args` that has model construction arguments.
 * @return struct llama_model_context* that is a model context. If it is unable to allocate model, it will return `NULL`.
 */
struct llama_model_context* llama_create_context(struct llama_model_context_args args);

/**
 * @brief Loads the model into memory
 * 
 * @param model_context is the context that is constructed using `llama_create_context`.
 * @param kind is the model id as `enum`.
 * @param filepath is the path to the model.
 * @return true if load is successful.
 * @return false if load is unsuccessful.
 */
bool llama_load_model(struct llama_model_context* model_context, enum ModelKind kind, char const* filepath);

/**
 * @brief Loads the model into memory
 * 
 * @param model_context is the context that is constructed using `llama_create_context`.
 * @param model_id is the model id as `c string`.
 * @param filepath is the path to the model.
 * @return true if load is successful.
 * @return false if load is unsuccessful.
 */
bool llama_load_model_str(struct llama_model_context* model_context, char const* model_id, char const* filepath);

/**
 * @brief Sets the stop words. It will overwrite the stop words with every call.
 * 
 * @param model_context is the context that is constructed using `llama_create_context`.
 * @param words is the array of `c strings`.
 * @param len is the size of the array.
 * @return true if it successfully sets the words.
 * @return false if it is unable to sets the words.
 */
bool llama_set_stop_words(struct llama_model_context* model_context, char const** words, size_t len);

/**
 * @brief Ingests the system prompt that will be preserved across memory reset to save memory.
 *        It is used to set the model behaviour.
 * 
 * @param model_context is the context that is constructed using `llama_create_context`.
 * @param prompt is user string that will be processed and produce output.
 * @return true if it successfully ingests the prompt.
 * @return false if it is unable to ingest the prompt.
 */
bool llama_ingest_system_prompt(struct llama_model_context* model_context, char const* prompt);

/**
 * @brief Ingests the prompt that will not be preserved across memory reset to save memory.
 *        It is used for having conversation with the model.
 * 
 * @param model_context is the context that is constructed using `llama_create_context`.
 * @param prompt is user string that will be processed and produce output.
 * @return true if it successfully ingests the prompt.
 * @return false if it is unable to ingest the prompt.
 */
bool llama_ingest(struct llama_model_context* model_context, char const* prompt);

/**
 * @brief Generate the model output from pervious ingested prompt or past conversation.
 *        It evaluates the model.
 * 
 * @param model_context is the context that is constructed using `llama_create_context`.
 * @param stream_fn is the callback function that is called every time model generates a token.
 * @param number_of_tokens is the maximum number of token that the model can generate.
 * @param top_k controls the diversity by limiting the selection to the top k highest probability tokens.
 * @param top_p filters out tokens based on cumulative probability, further refining diversity.
 * @param temp adjusts the sampling temperature, influencing creativity and randomness.
 * @param repeat_penalty penalizes repeated tokens to reduce redundancy in generated text.
 * @return true if it generates the output without any hitch.
 * @return false if it encounters an error.
 */
bool llama_generate(
    struct llama_model_context* model_context,
    LLAMA_STREAM_FUNC stream_fn,
    size_t number_of_tokens,
    float top_k,
    float top_p,
    float temp,
    float repeat_penalty
);

/**
 * @brief This function calculates the perplexity of the model for a given prompt
 * 
 * @param ctx is a model context of type `llama_model_context`
 * @param prompt is a C string that contains user prompt
 * @return `llama_array_view` that will contain the perplexity but if it fails, it will return -1;
 */
float llama_perplexity(struct llama_model_context* ctx, char const* prompt);

/**
 * @brief Getter for getting the embedding from model context. It will return empty view if flag for getting
 *        embedding is not set.
 * 
 * @param model_context is the context that is constructed using `llama_create_context`.
 * @return struct llama_array_view_f that contains the pointer to the embedding array and size of the array
 */
struct llama_array_view_f llama_get_embeddings(struct llama_model_context const* const model_context);

/**
 * @brief Getter for getting the logits from model context. It will return all the logits when the flag `should_get_all_logits`
 *        is set. Otherwise, it will return the logits of the size equivalent to the vocabulary size.
 * 
 * @param model_context is the context that is constructed using `llama_create_context`.
 * @return struct llama_array_view_f that contains the pointer to the logits array and size of the array
 */
struct llama_array_view_f llama_get_logits(struct llama_model_context const* const model_context);

/**
 * @brief Saves the model state to the file path.
 * 
 * @param model_context is the context that is constructed using `llama_create_context`.
 * @param filepath is the path to the file where the model state will be saved.
 * @return true if it successfully saves the model state.
 */
bool llama_save_state(struct llama_model_context* model_context, char const* filepath);

/**
 * @brief Loads the model state from the file path.
 * 
 * @param model_context is the context that is constructed using `llama_create_context`.
 * @param filepath is the path to the file where the model state will be loaded.
 * @return true if it successfully loads the model state.
 */
bool llama_load_state(struct llama_model_context* model_context, char const* filepath);

/**
 * @brief Allows to add lora adapter to the model context.
 * 
 * @param model_context is the context that is constructed using `llama_create_context`.
 * @param filepath is the path to the lora adapter.
 * @return true if it successfully loads the lora adapter.
 * @return false if it fails to load the lora adapter.
 */
bool llama_attach_lora(struct llama_model_context* model_context, char const* filepath);

/**
 * @brief Removes the lora adapter from the model context.
 * 
 * @param model_context is the context that is constructed using `llama_create_context`.
 * @return true if it successfully removes the lora adapter.
 * @return false if it fails to remove the lora adapter.
 */
bool llama_detach_lora(struct llama_model_context* model_context);

/**
 * @brief Resets the model context. It will reset the model state and the memory.
 * 
 * @param model_context is the context that is constructed using `llama_create_context`.
 * @return true if it successfully resets the model context.
 * @return false if it fails to reset the model context.
 */
bool llama_reset_model(struct llama_model_context* model_context);

/**
 * @brief Frees the model context.
 * 
 */
void llama_free_context(struct llama_model_context*);

#ifdef __cplusplus
}
#endif

#endif // FAST_LLAMA_BRIDGE_H
