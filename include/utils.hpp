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

    } // namespace literals
    

} // namespace fastllama


#endif // FAST_LLAMA_UTILS_HPP
