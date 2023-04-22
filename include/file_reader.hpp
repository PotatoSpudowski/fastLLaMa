#if !defined(FAST_LLAMA_FILE_READER_HPP)
#define FAST_LLAMA_FILE_READER_HPP

#include <string>
#include <type_traits>
#include "detail/file.hpp"

namespace fastllama{
    
    struct BinaryFileReader : detail::File {

        BinaryFileReader(std::string_view path) noexcept
            : File(path.data(), "rb")
        {}
        
        BinaryFileReader(BinaryFileReader const& other) noexcept = delete;
        BinaryFileReader(BinaryFileReader&& other) noexcept = default;
        BinaryFileReader& operator=(BinaryFileReader&& other) noexcept = default;
        BinaryFileReader& operator=(BinaryFileReader const& other) noexcept = delete;
        ~BinaryFileReader() noexcept = default;

        template<typename T>
        auto read(T* val, std::size_t num_of_objects = 1) noexcept -> bool {
            auto const size = sizeof(std::decay_t<T>);
            auto read_count = std::fread(reinterpret_cast<void *>(val), size, num_of_objects, handle());
            return read_count == num_of_objects;
        }
        
        auto read(void* val, std::size_t type_size_in_bytes, std::size_t num_of_objects = 1) noexcept -> bool {
            auto const size = type_size_in_bytes;
            auto read_count = std::fread(reinterpret_cast<void *>(val), size, num_of_objects, handle());
            return read_count == num_of_objects;
        }

        auto read_u8() noexcept -> std::uint8_t {
            std::uint8_t val;
            auto const res = read(&val);
            FAST_LLAMA_ASSERT(res, "read_u8 failed");
            return val;
        }
        
        auto read_bool() noexcept -> bool {
            bool val;
            auto const res = read(&val);
            FAST_LLAMA_ASSERT(res, "read_bool failed");
            return val;
        }

        auto read_u32() noexcept -> std::uint32_t {
            std::uint32_t val;
            auto const res = read(&val);
            FAST_LLAMA_ASSERT(res, "read_u32 failed");
            return val;
        }

        auto read_f32() noexcept -> float {
            float val;
            auto const res = read(&val);
            FAST_LLAMA_ASSERT(res, "read_f32 failed");
            return val;
        }

        auto read_string(std::size_t max_size) noexcept -> std::string {
            std::string str(max_size, '\0');
            auto const res = read(str.data(), max_size);
            FAST_LLAMA_ASSERT(res, "read_string failed");
            return str;
        }

        template<typename SizeType = std::uint32_t>
        auto read_string() noexcept -> std::string {
            auto size = SizeType{};
            auto const res = read(&size);
            FAST_LLAMA_ASSERT(res, "read_string failed");
            return read_string(static_cast<std::size_t>(size));
        }
    };

} // namespace fastllama


#endif // FAST_LLAMA_FILE_READER_HPP
