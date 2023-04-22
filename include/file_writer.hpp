#if !defined(FAST_LLAMA_FILE_WRITER_HPP)
#define FAST_LLAMA_FILE_WRITER_HPP

#include <string>
#include <type_traits>
#include "detail/file.hpp"

namespace fastllama{
    
    struct BinaryFileWriter: detail::File {

        BinaryFileWriter(std::string_view path) noexcept
            : File(path.data(), "wb")
        {}
        
        BinaryFileWriter(BinaryFileWriter const& other) noexcept = delete;

        BinaryFileWriter(BinaryFileWriter&& other) noexcept = default;
        
        BinaryFileWriter& operator=(BinaryFileWriter&& other) noexcept = default;
        
        BinaryFileWriter& operator=(BinaryFileWriter const& other) noexcept = delete;

        template<typename T>
        auto write(T const* val, std::size_t num_of_objects = 1) noexcept -> bool {
            auto const size = sizeof(std::decay_t<T>);
            auto read_count = std::fwrite(reinterpret_cast<void const*>(val), size, num_of_objects, handle());
            return read_count == num_of_objects;
        }
        
        auto write(void const* val, std::size_t type_size_in_bytes, std::size_t num_of_objects = 1) noexcept -> bool {
            auto const size = type_size_in_bytes;
            auto read_count = std::fwrite(reinterpret_cast<void const*>(val), size, num_of_objects, handle());
            return read_count == num_of_objects;
        }

        auto write_u8(std::uint8_t val) noexcept -> bool {
            return write(&val);
        }

        auto write_bool(bool val) noexcept -> bool {
            return write(&val);
        }
        
        auto write_u32(std::uint32_t val) noexcept -> bool {
            return write(&val);
        }

        auto write_f32(float val) noexcept -> bool {
            return write(&val);
        }

        auto write_string(std::string_view str) noexcept -> bool {
            auto const size = str.size();
            auto const res = write_u32(static_cast<std::uint32_t>(size));
            if (!res) return false;
            return write(str.data(), size);
        }
    };

} // namespace fastllama


#endif // FAST_LLAMA_FILE_WRITER_HPP
