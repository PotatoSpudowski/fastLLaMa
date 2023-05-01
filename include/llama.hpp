#if !defined(FAST_LLAMA_LLAMA_HPP)
#define FAST_LLAMA_LLAMA_HPP

#include "ggml.h"
#include "vocab.hpp"
#include "model_type.hpp"
#include <cstdint>
#include <vector>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include "logger.hpp"
#include "span.hpp"
#include "file_writer.hpp"
#include "mmap.hpp"
#include "file_reader.hpp"
#include "uninitialized_buffer.hpp"
#include "tensor/mem_context.hpp"
#include "tensor/utils.hpp"

namespace fastllama {

    enum class FType {
        ALL_F32     = 0,
        MOSTLY_F16  = 1,  // except 1d tensors
        MOSTLY_Q4_0 = 2,  // except 1d tensors
        MOSTLY_Q4_1 = 3,  // except 1d tensors
        MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        MOSTLY_Q4_2 = 5,  // except 1d tensors
        MOSTLY_Q4_3 = 6,  // except 1d tensors
        SIZE
    };

    enum class MagicKind : std::uint32_t {
        Unknown = 0,
        GGML = 0x67676d6c,
        GGMF = 0x67676d66,
        GGLA = 0x67676c61,
        GGJT = 0x67676a74,
    };


    enum class FileVersion : std::uint32_t {
        GGML,
        GGMF_V1,   // added version field and scores in vocab
        GGJT_V1,   // added padding
        Size
    };
    
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

    struct LoraAdapterParams {
        std::uint32_t r{1};
        std::uint32_t alpha{1};
        bool use_cache_matrix{false};

        constexpr float get_scale() const noexcept {
            return static_cast<float>(alpha) / static_cast<float>(r);
        }
    };

    struct HyperParams {
        std::uint32_t n_vocab { 32000 };
        std::uint32_t n_ctx   { 512 };
        std::uint32_t n_embd  { 4096 };
        std::uint32_t n_mult  { 256 };
        std::uint32_t n_head  { 32 };
        std::uint32_t n_layer { 32 };
        std::uint32_t n_rot   { 64 };
        FType        ftype   { FType::MOSTLY_F16 };

        constexpr bool operator==(HyperParams const& other) const noexcept {
            return n_vocab == other.n_vocab &&
                   n_ctx   == other.n_ctx   &&
                   n_embd  == other.n_embd  &&
                   n_mult  == other.n_mult  &&
                   n_head  == other.n_head  &&
                   n_layer == other.n_layer &&
                   n_rot   == other.n_rot   &&
                   ftype   == other.ftype;
        }
        constexpr bool operator!=(HyperParams const& other) const noexcept {
            return !(*this == other);
        }
    };

    struct KVCacheBuffer {

        bool init(HyperParams const& params, Logger const& logger = Logger{});
        void deinit(Logger const& logger = Logger{});
        bool save_state(BinaryFileWriter& writer, Logger const& logger) const noexcept;
        bool load_state(BinaryFileReader& reader, Logger const& logger) noexcept;

        ggml_type memory_type{ GGML_TYPE_F32 };

        // key + value memory
        ggml_tensor * k;
        ggml_tensor * v;

        MemContext ctx;

        UninitializedBuffer buffer;

        std::size_t number_of_tokens_in_cache{};
    };

    struct Model {
        using vocab_id = typename Vocab::id_type;

        static constexpr std::size_t max_number_of_scratch_buffer = 16;
    #ifdef LLAMA_USE_SCRATCH
        static constexpr bool use_scratch_buffer = true;
    #else
        static constexpr bool use_scratch_buffer = false;
    #endif

        bool load(std::string_view filepath);
        auto unload() -> void;
        auto eval(
            std::size_t                     n_past,
            Span<vocab_id>                  embd_inp,
            std::vector<float>&             embd_w,
            std::size_t&                    mem_per_token
        ) -> bool;

        auto set_threads(int in_threads) noexcept {
            this->threads = std::max(1, std::min(static_cast<int>(std::thread::hardware_concurrency()), in_threads));
        }

        bool dump_vocab(std::string_view filepath);
        bool attach_lora(std::string_view filepath);
        bool detach_lora();

        void use_buf([[maybe_unused]] ggml_context* in_ctx, [[maybe_unused]] int i) {
            if constexpr (use_scratch_buffer) {
                auto last_size = std::size_t{};

                if (i == -1) {
                    last_size = ggml_set_scratch(in_ctx, { 0, 0, nullptr });
                } else {
                    auto& buff = buf_scratch[i];
                    last_size = ggml_set_scratch(in_ctx, { 0, buff.size(), buff.data() });
                }

                if (last_size > 0) buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
                
                buf_last = i;
            }
        }

        std::size_t get_buf_max_mem(int i) {
            if constexpr(use_scratch_buffer) {
                return buf_max_size[i];
            } else {
                return 0;
            }
        }

        bool save_state(BinaryFileWriter& writer) const noexcept;
        bool load_state(BinaryFileReader& reader) noexcept;

        bool reset() noexcept;

        Logger logger{};

        ModelId model_id{};
        Vocab vocabulary;

        HyperParams params;

        ggml_tensor* tok_embeddings;
        
        ggml_tensor * norm;
        ggml_tensor * output;

        std::vector<Layer> layers;
        std::vector<float> embeddings;

        MemContext ctx;
        UninitializedBuffer buffer;

        KVCacheBuffer kv_self{};

        UninitializedBuffer buf_compute;
        UninitializedBuffer buf_scratch[max_number_of_scratch_buffer];

        int    buf_last = 0;
        std::size_t buf_max_size[max_number_of_scratch_buffer] = { 0 };
        std::size_t allocate_extra_mem{};

        MemoryLock mlock_buffer;
        MemoryLock mlock_mmap;

        // The current position in bump allocator.
        std::size_t buffer_lora_head{};
        // Memory for bump allocator
        UninitializedBuffer buffer_lora_for_mmap;
        std::unordered_map<std::string, void*> org_tensor_data_ptr_for_mmap;

        TensorsMapping tensors;

        std::string attached_lora_path{};

        std::unordered_map<std::string, ggml_tensor*> tensor_by_name;

        std::unique_ptr<MMappedFile> mapping;

        bool            is_valid{false};
        bool            embeddings_eval_enable{false};
        bool            should_put_all_logits{false};
        bool            use_mmap{false};
        bool            use_mlock{false};
        bool            load_parallel{false};
        int             threads{ static_cast<int>(std::thread::hardware_concurrency()) };
        int             n_batch{64};
        std::uint32_t   n_load_parallel_blocks{1}; // Block size for parallel loading
        FileVersion     file_version{ FileVersion::GGML };
    };

    bool quantize(std::string_view in_filepath, std::string_view out_filepath, FType ftype, int threads);

    constexpr std::string_view to_string_view(FType ftype) noexcept {
        switch (ftype) {
            case fastllama::FType::ALL_F32: return "all F32";
            case fastllama::FType::MOSTLY_F16: return "mostly F16";
            case fastllama::FType::MOSTLY_Q4_0: return "mostly Q4_0";
            case fastllama::FType::MOSTLY_Q4_1: return "mostly Q4_1";
            case fastllama::FType::MOSTLY_Q4_2: return "mostly Q4_2";
            case fastllama::FType::MOSTLY_Q4_3: return "mostly Q4_3";
            case fastllama::FType::MOSTLY_Q4_1_SOME_F16: return "mostly Q4_1, some F16";
            default: return "unknown, may not work";
        }
    }

} // namespace fastllama

#endif // FAST_LLAMA_LLAMA_HPP

