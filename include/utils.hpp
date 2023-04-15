#if !defined(FAST_LLAMA_UTILS_HPP)
#define FAST_LLAMA_UTILS_HPP

#include <cstddef>

namespace fastllama {
    
    namespace literals {
        
        constexpr auto operator""_GiB (unsigned long long int  size) noexcept {
            return size * (1024 * 1024 * 1024);
        }

        constexpr auto operator""_MiB (unsigned long long int  size) noexcept {
            return size * (1024 * 1024);
        }
        
        constexpr auto operator""_KiB (unsigned long long int  size) noexcept {
            return size * 1024;
        }
        
        constexpr auto operator""_GiB (long double  size) noexcept {
            return size * static_cast<long double>(1024.0 * 1024.0 * 1024.0);
        }

        constexpr auto operator""_MiB (long double  size) noexcept {
            return size * static_cast<long double>(1024.0 * 1024.0);
        }
        
        constexpr auto operator""_KiB (long double  size) noexcept {
            return size * static_cast<long double>(1024.0);
        }

    } // namespace literals
    

} // namespace fastllama


#endif // FAST_LLAMA_UTILS_HPP
