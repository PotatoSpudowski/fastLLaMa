#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fastllama.h"

void stream_fn(char const* token_stream, int len) {
    printf("%.*s", len, token_stream);
    fflush(stdout);
}

int main() {
    struct llama_model_context_args args = llama_create_default_context_args();
    args.n_threads = 16;
    args.n_batch = 64;
    args.n_ctx = 512;
    args.n_keep = 48;

    struct llama_model_context* model_ctx = llama_create_context(args);
    
    if (!llama_load_model(model_ctx, ALPACA_LORA_7B, "./models/ALPACA-LORA-7B/alpaca-lora-q4_0.bin")) {
        return 1;
    }

    char const prefix[] = "\n\n### Instruction:\n\n";
    char const suffix[] = "\n\n### Response:\n\n";
    size_t const prompt_size = 1024 + sizeof(prefix);

    char* prompt = calloc(prompt_size + sizeof(prefix) + sizeof(suffix), sizeof(char));
    strcpy(prompt, prefix);

    printf("User: ");

    char const* stop_words[] = {
        "###",
    };
    llama_set_stop_words(model_ctx, stop_words, sizeof(stop_words)/sizeof(stop_words[0]));

    while(true) {
        char* res = fgets(prompt + sizeof(prefix) - 1, prompt_size - 1, stdin);
        (void)res;
        size_t str_len = strlen(prompt);
        strcpy(prompt + str_len - 1, suffix);
        prompt[str_len + sizeof(suffix) - 1] = 0;

        if (!llama_ingest(model_ctx, prompt)) {
            return 2;
        }

        if (!llama_generate(model_ctx, &stream_fn, 300, 40, 0.95f, 0.8f, 1.0f)) {
            return 3;
        }

        printf("\nUser: ");
    }

    free(prompt);
    
    return 0;
}