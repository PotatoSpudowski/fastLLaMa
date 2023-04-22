#include "llama.hpp"

// usage:
//  ./llama-quantize models/llama/ggml-model.bin models/llama/ggml-model-quant.bin type
//
int main(int argc, char ** argv) {
    ggml_time_init();

    if (argc != 4) {
        fprintf(stderr, "usage: %s model-f32.bin model-quant.bin type\n", argv[0]);
        fprintf(stderr, "  type = 2 - q4_0\n");
        fprintf(stderr, "  type = 3 - q4_1\n");
        return 1;
    }

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    std::string_view fname_inp = argv[1];
    std::string_view fname_out = argv[2];

    auto const itype = static_cast<fastllama::FType>(atoi(argv[3]));

    auto const t_main_start_us = ggml_time_us();

    std::int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!fastllama::quantize(fname_inp, fname_out, itype, 8)) {
            fprintf(stderr, "%s: failed to quantize model from '%.*s'\n", __func__, static_cast<int>(fname_inp.size()), fname_inp.data());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0);
    }

    return 0;
}
