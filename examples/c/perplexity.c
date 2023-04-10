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
    args.n_keep = 200;
    args.allocate_extra_mem = 512 * (20ul << 20); // Adding extra memory for evaluation

    struct llama_model_context* model_ctx = llama_create_context(args);
    
    if (!llama_load_model(model_ctx, LLAMA_7B, "./models/7B/ggml-model-q4_0.bin")) {
        return 1;
    }

    // You can use your own text to use, or you can use the following file
    // 1. https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip\?ref\=salesforce-research
    // 2. unzip the file
    // 3. get the path to the `wiki.test.raw` file.
    FILE* file = fopen("./wikitext-2-raw/wiki.test.raw", "r");

    // Get 8000 characters for testing
    char buffer[8000] = {0};
    fread(&buffer, sizeof(char), sizeof(buffer) - 1, file);

    float final_result = llama_perplexity(model_ctx, buffer);
    printf("Total Perplexity: %.4f", final_result);
    
    return 0;
}