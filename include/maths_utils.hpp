#if !defined(FAST_LLAMA_MATHS_UTILS_HPP)
#define FAST_LLAMA_MATHS_UTILS_HPP

#include <optional>

namespace fastllama {
    
    template<typename T>
    constexpr std::optional<T> checked_mul(T a, T b) noexcept {
        static_assert(std::is_integral_v<T>, "T must be an integral type");
        T result = a * b;
        if (a != 0 && result / a != b) return {};
        return result;
    }

    template<typename T>
    constexpr std::optional<T> checked_add(T a, T b) noexcept {
        static_assert(std::is_integral_v<T>, "T must be an integral type");
        T result = a + b;
        if (result < a) return {};
        return result;
    }

    template<typename T>
    constexpr std::optional<T> checked_sub(T a, T b) noexcept {
        static_assert(std::is_integral_v<T>, "T must be an integral type");
        T result = a - b;
        if (result > a) return {};
        return result;
    }

    template<typename T>
    constexpr std::optional<T> checked_div(T a, T b) noexcept {
        static_assert(std::is_integral_v<T>, "T must be an integral type");
        if (b == 0 || a % b != 0) return {};
        return a / b;
    }

} // namespace fastllama

#endif // FAST_LLAMA_MATHS_UTILS_HPP
