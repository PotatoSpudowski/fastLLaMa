#if !defined(FAST_LLAMA_SPAN_HPP)
#define FAST_LLAMA_SPAN_HPP

#include <vector>
#include <array>
#include <limits>

namespace fastllama {
    
    static constexpr std::size_t dynamic_v = std::numeric_limits<std::size_t>::max();

    template<typename T>
    struct Span {
        using value_type = std::decay_t<std::remove_cv_t<T>>;
        using pointer = std::add_pointer_t<std::add_const_t<value_type>>;
        using reference = std::add_rvalue_reference_t<std::add_const_t<value_type>>;
        using size_type = std::size_t;
        using iterator = pointer;
        
        constexpr Span(std::vector<T> const& v) noexcept
            : m_data(v.data())
            , m_size(v.size())
        {}
        
        template<std::size_t N>
        constexpr Span(std::array<T, N> const& v) noexcept
            : m_data(v.data())
            , m_size(v.size())
        {}
        
        template<std::size_t N>
        constexpr Span(T const (& v)[N]) noexcept
            : m_data(&v[0])
            , m_size(N)
        {}
        
        constexpr Span(T const* data, size_type size) noexcept
            : m_data(data)
            , m_size(size)
        {}

        constexpr Span() noexcept = default;
        constexpr Span(Span const&) noexcept = default;
        constexpr Span(Span &&) noexcept = default;
        constexpr Span& operator=(Span const&) noexcept = default;
        constexpr Span& operator=(Span &&) noexcept = default;
        ~Span() noexcept = default;

        constexpr bool empty() const noexcept { return (m_size == 0ul) || (m_data == nullptr); }
        constexpr size_type size() const noexcept { return m_size; }
        constexpr pointer data() const noexcept { return m_data; }

        constexpr iterator begin() const noexcept { return m_data; }
        constexpr iterator end() const noexcept { return m_data + m_size; }

        constexpr reference operator[](size_type k) const noexcept { return m_data[k]; }

        constexpr Span sub_view(size_type start, size_type end = dynamic_v) const noexcept {
            return { m_data + start, std::min(m_size, end) };
        }

    private:
        pointer     m_data{nullptr};
        size_type   m_size{};
    };

} // namespace fastllama


#endif // FAST_LLAMA_SPAN_HPP
