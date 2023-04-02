#if !defined(FAST_LLAMA_LLAMA_HPP)
#define FAST_LLAMA_LLAMA_HPP

#include "ggml.h"
#include "vocab.hpp"
#include "model_type.hpp"
#include <cstdint>
#include <vector>
#include <unordered_map>
#include "logger.hpp"

namespace fastllama {

    inline static constexpr std::uint32_t magic_number_v = 0x67676d6c;
    
    struct Layer {
        // normalization
        ggml_tensor * attention_norm;

        // attention
        ggml_tensor * wq;
        ggml_tensor * wk;
        ggml_tensor * wv;
        ggml_tensor * wo;

        // normalization
        ggml_tensor * ffn_norm;

        // ff
        ggml_tensor * w1;
        ggml_tensor * w2;
        ggml_tensor * w3;
    };

    struct HyperParams {
        std::int32_t n_vocab { 32000 };
        std::int32_t n_ctx   { 512 };
        std::int32_t n_embd  { 4096 };
        std::int32_t n_mult  { 256 };
        std::int32_t n_head  { 32 };
        std::int32_t n_layer { 32 };
        std::int32_t n_rot   { 64 };
        std::int32_t f16     { 1 };
    };

    struct Model {
        using vocab_id = typename Vocab::id;

        bool load(std::string_view model_name, std::string_view filepath);
        auto unload();
        auto eval(
            int threads,
            std::size_t n_past,
            std::vector<vocab_id> const& embd_inp,
            std::vector<float>& embd_w,
            std::size_t& mem_per_token
        ) -> bool;

        Logger logger{};

        ModelId model_id{};
        Vocab vocabulary;

        HyperParams params;

        ggml_tensor* tok_embeddings;
        
        ggml_tensor * norm;
        ggml_tensor * output;

        std::vector<Layer> layers;

        // key + value memory
        ggml_tensor * memory_k;
        ggml_tensor * memory_v;

        ggml_context * ctx;
        std::unordered_map<std::string, ggml_tensor*> tensors;

        bool is_valid{false};

    private:
        std::size_t m_buffer_size{ 512 * 1024 * 1024 };
        std::vector<unsigned char> m_buffer;
    };

} // namespace fastllama


#endif // FAST_LLAMA_LLAMA_HPP

