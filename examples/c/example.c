#include <stdio.h>
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
    args.n_keep = 200;

    struct llama_model_context* model_ctx = llama_create_context(args);
    
    if (!llama_load_model(model_ctx, LLAMA_7B, "./models/7B/ggml-model-q4_0.bin")) {
        return 1;
    }

    char const* system_prompt = "Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.\n"
        "User: Hello, Bob.\n"
        "Bob: Hello. How may I help you today?\n"
        "User: Please tell me the largest city in Europe.\n"
        "Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.\n"
        "User: ";

    printf("Ingesting, please wait...\n");
    fflush(stdout);

    if (!llama_ingest_system_prompt(model_ctx, system_prompt)) {
        return 2;
    }

    printf("Ingestion complete!\n");
    fflush(stdout);

    char prompt[256] = "User: ";

    printf("User: ");

    llama_set_stop_words(model_ctx, 1, "User:");

    while(true) {
        int len = scanf("%s", prompt + 6);
        if (len < 0) break;

        prompt[len] = 0;
        
        if (!llama_ingest(model_ctx, prompt)) {
            return 2;
        }

        if (!llama_generate(model_ctx, &stream_fn, 300, 40, 0.95f, 0.8f, 1.0f)) {
            return 3;
        }

        
        printf("\n\nnUser: ");
    }
    
    return 0;
}