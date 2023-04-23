#if !defined(FAST_LLAMA_MEM_CONTEXT_HPP)
#define FAST_LLAMA_MEM_CONTEXT_HPP

#include "ggml.h"
#include "uninitialized_buffer.hpp"

namespace fastllama {
    
    struct MemContext {
        using size_type = std::size_t;

        MemContext(void* buff, size_type size, bool no_alloc = false) noexcept {
            ggml_init_params params{};
            params.mem_size = size;
            params.mem_buffer = buff;
            params.no_alloc = no_alloc;
            ctx = ggml_init(params);
        }
        
        MemContext(size_type size) noexcept {
            ggml_init_params params{};
            params.mem_size = size;
            ctx = ggml_init(params);
        }
        
        MemContext(UninitializedBuffer& buff, bool no_alloc = false)
            : MemContext(buff.data(), buff.size(), no_alloc)
        {}

        MemContext() = default;
        MemContext(MemContext const&) = default;
        MemContext(MemContext&&) noexcept = default;
        MemContext& operator=(MemContext const&) = default;
        MemContext& operator=(MemContext&&) noexcept = default;

        void free() noexcept {
            if (ctx) ggml_free(ctx);
            ctx = nullptr;
        }

        ~MemContext() noexcept {
            free();
        }

        constexpr auto get() const noexcept -> ggml_context* {
            return ctx;
        }

        constexpr auto get() noexcept -> ggml_context* {
            return ctx;
        }

        constexpr operator bool() const noexcept {
            return ctx != nullptr;
        }

        constexpr operator ggml_context* () noexcept {
            return ctx;
        }

    private:
        ggml_context* ctx{nullptr};
    };

} // namespace fastllama


#endif // FAST_LLAMA_MEM_CONTEXT_HPP
