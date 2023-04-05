#if !defined(FAST_LLAMA_FILE_PIPE_HPP)
#define FAST_LLAMA_FILE_PIPE_HPP

#include <string>
#include <cstdio>
#include <optional>
#include <type_traits>
#include "file_writer.hpp"
#include "file_reader.hpp"

namespace fastllama {

    namespace detail {
        
        template<typename T>
        struct IdentityFn {
            constexpr auto operator()(T* in) const noexcept { return in; }
        };

    } // namespace detail
    
    
    struct BinaryFilePipe {
        constexpr BinaryFilePipe(BinaryFileReader& reader, BinaryFileWriter& writer) noexcept
            : m_reader(std::move(reader))
            , m_writer(std::move(writer))
        {}
        
        constexpr BinaryFilePipe(std::string_view reader_filepath, std::string_view& writer_filepath) noexcept
            : m_reader(reader_filepath)
            , m_writer(writer_filepath)
        {}

        BinaryFilePipe(BinaryFilePipe const&) noexcept = default;
        BinaryFilePipe(BinaryFilePipe &&) noexcept = default;
        BinaryFilePipe& operator=(BinaryFilePipe const&) noexcept = default;
        BinaryFilePipe& operator=(BinaryFilePipe &&) noexcept = default;
        ~BinaryFilePipe() noexcept = default;

        template<typename T, typename Fn = detail::IdentityFn<T>>
        auto read_and_write(T* val, std::size_t num_of_objects = 1, Fn&& fn = Fn{}) noexcept -> bool {
            if (!m_reader.read(val, num_of_objects)) return false;
            auto* new_val = fn(val);
            return m_writer.write(val, num_of_objects);
        }
        
        template<typename Fn = detail::IdentityFn<void>>
        auto read_and_write(void* val, std::size_t type_size_in_bytes, std::size_t num_of_objects = 1, Fn&& fn = Fn{}) noexcept -> bool {
            if (!m_reader.read(val, type_size_in_bytes, num_of_objects)) return false;
            auto* new_val = fn(val);
            return m_writer.write(val, type_size_in_bytes, num_of_objects);
        }

        constexpr auto& get_reader() const noexcept { return m_reader; }
        constexpr auto& get_writer() const noexcept { return m_writer; }

        constexpr auto& get_reader() noexcept { return m_reader; }
        constexpr auto& get_writer() noexcept { return m_writer; }

        constexpr operator bool() const noexcept {
            return m_reader && m_writer;
        }

        constexpr void close() noexcept {
            m_reader.close();
            m_writer.close();
        }

    private:
        BinaryFileReader m_reader;
        BinaryFileWriter m_writer;
    };

} // namespace fastllama


#endif // FAST_LLAMA_FILE_PIPE_HPP
