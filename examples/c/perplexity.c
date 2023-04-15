#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fastllama.h"

int main() {
    struct llama_model_context_args args = llama_create_default_context_args();
    args.n_threads = 16;
    args.n_batch = 512; // number of samples per batch
    args.n_ctx = 512;
    args.n_keep = 200;

    struct llama_model_context* model_ctx = llama_create_context(args);
    
    if (!llama_load_model(model_ctx, LLAMA_7B, "./models/7B/ggml-model-q4_0.bin")) {
        return 1;
    }

    // Use some text file for testing that has at least 8000 characters
    FILE* file = fopen("./test.txt", "r");
    if (file == NULL) {
        printf("Failed to open file");
        return 1;
    }

    // Get 8000 characters for testing
    char buffer[8000] = {0};
    int read_count = fread(&buffer, sizeof(char), sizeof(buffer) - 1, file);
    if (read_count == 0) {
        printf("Failed to read file");
        return 1;
    }

    float final_result = llama_perplexity(model_ctx, buffer);
    printf("Total Perplexity: %.4f", final_result);
    
    return 0;
}