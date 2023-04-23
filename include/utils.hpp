#if !defined(FAST_LLAMA_UTILS_HPP)
#define FAST_LLAMA_UTILS_HPP

#include <cstddef>
#include <cstring>

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
    
        namespace detail {
        
        #if defined(_WIN32)
            inline static std::string llama_format_win_err(DWORD err) noexcept {
                LPSTR buf;
                size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                            NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
                if (!size) {
                    return "FormatMessageA failed";
                }
                std::string ret(buf, size);
                LocalFree(buf);
                return ret;
            }
        #endif
    } // namespace detail
    
    inline static std::string error_message() {
        #if defined(_WIN32)
            return detail::llama_format_win_err(GetLastError());
        #elif defined(__linux__)
            return std::strerror(errno);
        #else
            return "unknown error";
        #endif
    }

    inline static std::string_view humanize_size(char* buff, std::size_t n, std::size_t bytes) noexcept {

        constexpr std::size_t KiB = 1024;
        constexpr std::size_t MiB = KiB * 1024;
        constexpr std::size_t GiB = MiB * 1024;

        if (bytes < KiB) {
            std::snprintf(buff, n, "%zu B", bytes);
        } else if (bytes < MiB) {
            std::snprintf(buff, n, "%.2f KiB", static_cast<double>(bytes) / KiB);
        } else if (bytes < GiB) {
            std::snprintf(buff, n, "%.2f MiB", static_cast<double>(bytes) / MiB);
        } else {
            std::snprintf(buff, n, "%.2f GiB", static_cast<double>(bytes) / GiB);
        }

        return buff;
    }

    template<std::size_t N>
    inline static std::string_view humanize_size(char (&buff)[N], std::size_t bytes) noexcept {
        return humanize_size(buff, N, bytes);
    }

    inline static std::string dyn_humanize_size(std::size_t bytes, std::size_t max_size = 32) noexcept {
        std::string buff(max_size, '\0');
        auto size = humanize_size(buff.data(), buff.size(), bytes);
        buff.resize(size.size());
        return buff;
    }

} // namespace fastllama


#endif // FAST_LLAMA_UTILS_HPP
