#if !defined(FAST_LLAMA_RING_BUFFER_HPP)
#define FAST_LLAMA_RING_BUFFER_HPP

#include <type_traits>
#include <vector>
#include <cassert>
#include <iostream>

namespace fastllama {
    
    template<typename T>
    struct RingBuffer {
        using base_type = std::deque<T>;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using iterator =  typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;
        using size_type = typename base_type::size_type;
        using difference_type = typename base_type::difference_type;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using reverse_iterator = typename base_type::reverse_iterator;
        using const_reverse_iterator = typename base_type::const_reverse_iterator;

        RingBuffer(size_type size)
            : m_data(size)
            , m_capacity(size)
        {}

        RingBuffer(RingBuffer const&) = default;
        RingBuffer(RingBuffer &&) = default;
        RingBuffer& operator=(RingBuffer const&) = default;
        RingBuffer& operator=(RingBuffer &&) = default;
        ~RingBuffer() = default;

        auto push_back(value_type const& val) -> void {
            if (m_data.size() >= m_capacity) m_data.pop_front();
            m_data.push_back(val);
        }

        constexpr auto size() const noexcept -> size_type {
            return m_data.size();
        }

        constexpr auto empty() const noexcept -> bool {
            return m_data.empty();
        }

        constexpr const_reference operator[](size_type i) const noexcept {
            return m_data[i];
        }
        
        constexpr reference operator[](size_type i) noexcept {
            return m_data[i];
        }

        iterator begin() noexcept { return m_data.begin(); }
        iterator end() noexcept { return m_data.end(); }
        
        const_iterator begin() const noexcept { return m_data.begin(); }
        const_iterator end() const noexcept { return m_data.end(); }

        void resize(std::size_t size) {
            m_data.resize(size);
            m_capacity = size;
        }

        void clear() {
            m_data.clear();
        }

    private:
        base_type m_data;
        size_type m_capacity{};
    };
} // namespace fastllama


#endif // FAST_LLAMA_RING_BUFFER_HPP
