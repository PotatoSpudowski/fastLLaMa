#if !defined(FAST_LLAMA_DETAIL_FILE_HPP)
#define FAST_LLAMA_DETAIL_FILE_HPP

#include <cstdio>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <climits>

#include <string>
#include <vector>
#include "macro.hpp"

namespace fastllama::detail {
    
    struct File {
        enum class SeekReference {
            Begin = SEEK_SET,
            Current = SEEK_CUR,
            End = SEEK_END
        };

        File(std::string_view path, const char* mode) noexcept
            : m_path(path)
            , m_file(std::fopen(m_path.c_str(), mode))
        {
            if (!m_file) return;
            seek(0, SeekReference::End);
            m_size = tell();
            seek(0, SeekReference::Begin);
        }

        File(File const& other) noexcept = delete;
        // File(File const& other) noexcept
        //     : m_path(other.m_path)
        //     , m_file(m_file)
        //     , m_size(other.m_size)
        // {}
        
        File(File&& other) noexcept
            : m_path(std::move(other.m_path))
            , m_file(std::move(other.m_file))
            , m_size(std::move(other.m_size))
        {}
        
        // File& operator=(File const& other) noexcept {
        //     auto temp = File(other);
        //     swap(temp, *this);
        //     return *this;
        // }
        
        File& operator=(File const& other) noexcept = delete;
        
        File& operator=(File&& other) noexcept {
            auto temp = File(std::move(other));
            swap(temp, *this);
            return *this;
        }

        constexpr auto path() const noexcept -> std::string_view const {
            return m_path;
        }

        auto seek(std::size_t offset, SeekReference whence = SeekReference::Current) noexcept -> void {
            #ifdef _WIN32
                auto res = _fseeki64(m_file, offset, static_cast<int>(ref));
            #else
                auto res = std::fseek(m_file, offset, static_cast<int>(whence));
            #endif

            FAST_LLAMA_ASSERT(res == 0, "seek failed");
        }

        auto tell() const noexcept -> std::size_t {
            #ifdef _WIN32
                auto res = _ftelli64(m_file);
            #else
                auto res = std::ftell(m_file);
            #endif

            FAST_LLAMA_ASSERT(res >= 0, "tell failed");

            return static_cast<std::size_t>(res);
        }

        friend void swap(File& lhs, File& rhs) noexcept {
            std::swap(lhs.m_path, rhs.m_path);
            std::swap(lhs.m_file, rhs.m_file);
            std::swap(lhs.m_size, rhs.m_size);
        }

        ~File() noexcept {
            if (m_file) {
                std::fclose(m_file);
            }
        }

        constexpr auto handle() noexcept -> FILE* {
            return m_file;
        }
        
        constexpr auto handle() const noexcept -> FILE* {
            return m_file;
        }

        constexpr auto size() const noexcept -> std::size_t {
            return m_size;
        }

        constexpr operator bool() const noexcept {
            return static_cast<bool>(m_file);
        }

        auto eof() const noexcept -> bool {
            return feof(handle());
        }

        auto close() noexcept -> void {
            if (m_file) fclose(m_file);
            m_file = nullptr;
        }

        auto set_buffer(char* buffer, std::size_t size) noexcept {
            setvbuf(handle(), buffer, _IOFBF, size);
        }

    private:
        std::string m_path;
        FILE*       m_file{nullptr};
        std::size_t m_size{0};
    };

} // namespace fastllama::detail


#endif // FAST_LLAMA_DETAIL_FILE_HPP
