#if !defined(FAST_LLAMA_UNINITIALIZED_BUFFER_HPP)
#define FAST_LLAMA_UNINITIALIZED_BUFFER_HPP

#include <memory>

namespace fastllama {

    namespace detail {
        struct default_delete {
            void operator()(void* ptr) const { 
                if (ptr) std::free(ptr);
            }
        };
    } // namespace detail
    
    struct UninitializedBuffer {
        
        UninitializedBuffer() noexcept = default;
        UninitializedBuffer(UninitializedBuffer const& other) noexcept = delete;
        UninitializedBuffer(UninitializedBuffer&& other) noexcept
            : m_data(std::move(other.m_data))
            , m_size(std::move(other.m_size))
        {}
        UninitializedBuffer& operator=(UninitializedBuffer const& other) noexcept = delete;
        UninitializedBuffer& operator=(UninitializedBuffer&& other) noexcept = default;
        ~UninitializedBuffer() noexcept = default;

        UninitializedBuffer(std::size_t size, std::size_t alignment = sizeof(max_align_t)) {
            resize(size, alignment);
        }

        void resize(std::size_t size, std::size_t alignment = sizeof(max_align_t)) {
            allocate(size, alignment);
        }

        std::uint8_t* data() noexcept { return m_data.get(); }
        std::uint8_t const* data() const noexcept { return m_data.get(); }
        constexpr std::size_t size() const noexcept { return m_size; }

        operator bool() const noexcept { return m_data != nullptr; }

        std::uint8_t& operator[](std::size_t index) noexcept { return m_data.get()[index]; }
        std::uint8_t const& operator[](std::size_t index) const noexcept { return m_data.get()[index]; }

        auto begin() noexcept { return m_data.get(); }
        auto begin() const noexcept { return m_data.get(); }

        auto end() noexcept { return m_data.get() + m_size; }
        auto end() const noexcept { return m_data.get() + m_size; }

        void free() noexcept {
            m_data.reset();
            m_size = 0;
        }
    
    private:

        void allocate(std::size_t size, std::size_t alignment) {
            auto ptr = std::aligned_alloc(sizeof(max_align_t), size);
            m_data.reset(static_cast<std::uint8_t*>(ptr));
            m_size = size;
        }

    private:
        std::unique_ptr<std::uint8_t, detail::default_delete>   m_data{nullptr};
        std::size_t                                             m_size{0};
    };

} // namespace fastllama


#endif // FAST_LLAMA_UNINITIALIZED_BUFFER_HPP
